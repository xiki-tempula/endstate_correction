import abc
from typing import List, Literal, Union

from openmm import Integrator, LangevinIntegrator, Platform
from openmm import unit
from openmm.app import (
    AmberPrmtopFile,
    CharmmPsfFile,
    NoCutoff,
    PME,
    Simulation,
    Topology,
)
from openmmml import MLPotential

from ..constant import check_implementation
from ..protocol import BSSProtocol
from ..topology import AMBERTopology, CHARMMTopology


class EndstateCorrectionBase(abc.ABC):
    def __init__(
        self,
        top: Union[AMBERTopology, CHARMMTopology],
        env: Literal["complex", "waterbox", "vacuum"],
        ml_atoms: List[int],
        protocol: BSSProtocol,
        potential: str = "ani2x",
        interpolate: bool = True,
    ):
        self.top = top
        self.protocol = protocol
        self.env = env
        self._mm_topology = None

        mm_system = self.createSystem(self.env)
        ml_potential = MLPotential(potential)
        ml_system = ml_potential.createMixedSystem(
            self.get_mm_topology(), mm_system, ml_atoms, interpolate=interpolate
        )
        integrator = self.get_integrator()
        _, platform = check_implementation()
        platform = Platform.getPlatformByName(platform)
        self.simulation = Simulation(
            self.get_mm_topology(), ml_system, integrator, platform=platform
        )

    def get_mm_topology(self) -> Topology:
        if self._mm_topology is None:
            self._mm_topology = self._get_mm_topology()
        return self._mm_topology

    @abc.abstractmethod
    def _get_mm_topology(self) -> Union[CharmmPsfFile, AmberPrmtopFile]:
        pass

    def get_integrator(self) -> Integrator:
        return LangevinIntegrator(
            self.protocol.temperature,
            self.protocol.collision_rate,
            self.protocol.timestep,
        )

    def createSystem(self, **kwargs):
        if self.env == "vacuum":
            mm_system = self.get_mm_topology().createSystem(
                nonbondedMethod=NoCutoff, **kwargs
            )
        else:
            mm_system = self.get_mm_topology().createSystem(
                nonbondedMethod=PME,
                nonbondedCutoff=self.protocol.rlist * unit.nanometers,
                **kwargs
            )
        return mm_system
