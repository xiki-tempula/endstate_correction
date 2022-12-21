import abc
import logging
import os
from typing import List, Literal, Union

from openmm import Integrator, LangevinIntegrator, OpenMMException, Platform
from openmm import unit
from openmm.app import (
    AmberInpcrdFile,
    AmberPrmtopFile,
    CharmmPsfFile,
    DCDReporter,
    NoCutoff,
    PDBFile,
    PME,
    Simulation,
    Topology,
)
from openmmml import MLPotential

from ..constant import check_implementation
from ..equ import generate_samples
from ..protocol import BSSProtocol
from ..topology import AMBERTopology, CHARMMTopology


class EndstateCorrectionBase(abc.ABC):
    def __init__(
        self,
        top: Union[AMBERTopology, CHARMMTopology],
        env: Literal["complex", "waterbox", "vacuum"],
        ml_atoms: List[int],
        protocol: BSSProtocol,
        name: str = "endstate_correction",
        work_dir: str = "./",
        potential: str = "ani2x",
        interpolate: bool = True,
    ):
        self.logger = logging.getLogger("EndstateCorrectionBase")
        self.top = top
        self.protocol = protocol
        self.env = env
        self.name = name
        self.work_dir = work_dir
        self._mm_topology = None
        self._mm_coordinate = None

        mm_system = self._createSystem()
        ml_potential = MLPotential(potential)
        ml_system = ml_potential.createMixedSystem(
            self.get_mm_topology().topology,
            mm_system,
            ml_atoms,
            interpolate=interpolate,
        )
        integrator = self.get_integrator()
        _, platform = check_implementation()
        platform = Platform.getPlatformByName(platform)
        self.simulation = Simulation(
            self.get_mm_topology().topology, ml_system, integrator, platform=platform
        )

    def get_mm_topology(self) -> Topology:
        if self._mm_topology is None:
            self._mm_topology = self._get_mm_topology()
        return self._mm_topology

    def get_mm_coordinate(self) -> Union[PDBFile, AmberInpcrdFile]:
        if self._mm_coordinate is None:
            self._mm_coordinate = self._get_mm_coordinate()
        return self._mm_coordinate

    @abc.abstractmethod
    def _get_mm_topology(self) -> Union[CharmmPsfFile, AmberPrmtopFile]:
        pass

    @abc.abstractmethod
    def _get_mm_coordinate(self) -> Union[PDBFile, AmberInpcrdFile]:
        pass

    def get_integrator(self) -> Integrator:
        return LangevinIntegrator(
            self.protocol.temperature * unit.kelvin,
            self.protocol.collision_rate / unit.picosecond,
            self.protocol.timestep * unit.femtoseconds,
        )

    def _createSystem(self, **kwargs):
        if self.env == "vacuum":
            mm_system = self.get_mm_topology().createSystem(
                nonbondedMethod=NoCutoff, **kwargs
            )
        else:
            mm_system = self.get_mm_topology().createSystem(
                nonbondedMethod=PME,
                nonbondedCutoff=self.protocol.rlist * unit.nanometers,
                **kwargs,
            )
        return mm_system

    def start(self):
        n_steps_per_sample = self.protocol.restart_interval

        n_samples = int(
            (
                (self.protocol.runtime * unit.nanoseconds)
                / (self.protocol.timestep * unit.femtoseconds)
            )
            / n_steps_per_sample
        )

        # path where samples should be stored (will be created if it doesn't exist)
        base = f"{self.work_dir}/equilibrium_samples/{self.name}"
        os.makedirs(base, exist_ok=True)
        # define lambda states
        lamb = self.protocol.lam["ml-lambda"]
        self.logger.info(f"{lamb=}")
        # define where to store samples
        trajectory_file = f"{base}/{self.name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}_{self.env}.dcd"
        self._traj_file = trajectory_file
        self.logger.info(f"Trajectory saved to: {trajectory_file}")
        # set lambda
        self.simulation.context.setParameter("lambda_interpolate", lamb)
        # set coordinates
        self.simulation.context.setPositions(self.get_mm_coordinate().positions)
        # try to set velocities using openMM, fall back to manual velocity seeding if it fails
        if not self.protocol.restart:
            try:
                self.simulation.context.setVelocitiesToTemperature(
                    self.protocol.temperature
                )
            except OpenMMException:
                from endstate_correction.equ import _seed_velocities, _get_masses

                self.simulation.context.setVelocities(
                    _seed_velocities(_get_masses(self.simulation.system))
                )
        # define DCDReporter
        self.simulation.reporters.append(
            DCDReporter(
                trajectory_file,
                n_steps_per_sample,
            )
        )
        # perform sampling
        samples = generate_samples(
            self.simulation, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
        )
        self.simulation.reporters.clear()
