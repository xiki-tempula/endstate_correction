# general imports
import json

import openmm as mm
from openmm import unit
from openmm.app import PME, CharmmParameterSet, CharmmPsfFile, NoCutoff, Simulation
from openmmml import MLPotential
from tqdm import tqdm

from endstate_correction.constant import (
    collision_rate,
    stepsize,
    temperature,
    check_implementation,
)


def read_box(psf: CharmmPsfFile, filename: str) -> CharmmPsfFile:
    """set waterbox dimensions given the sysinfo file provided by CHARMM-GUI

    Args:
        psf (CharmmPsfFile): topology instance
        filename (str): filename for sysinfo file

    Returns:
        CharmmPsfFile: topology instance with box dimensions set
    """
    try:
        sysinfo = json.load(open(filename, "r"))
        boxlx, boxly, boxlz = map(float, sysinfo["dimensions"][:3])
    except:
        for line in open(filename, "r"):
            segments = line.split("=")
            if segments[0].strip() == "BOXLX":
                boxlx = float(segments[1])
            if segments[0].strip() == "BOXLY":
                boxly = float(segments[1])
            if segments[0].strip() == "BOXLZ":
                boxlz = float(segments[1])
    psf.setBox(boxlx * unit.angstroms, boxly * unit.angstroms, boxlz * unit.angstroms)
    return psf


def create_charmm_system(
    psf: CharmmPsfFile,
    parameters: CharmmParameterSet,
    env: str,
    ml_atoms: list,
) -> Simulation:
    """Generate an openMM simulation object using CHARMM topology and parameter files

    Args:
        psf (CharmmPsfFile): topology instance
        parameters (CharmmParameterSet): parameter instance
        env (str): either complex, waterbox or vacuum
        ml_atoms (list): list of atoms described by the QML potential


    Returns:
        Simulation: openMM simulation instance
    """

    ###################
    print(f"Generating charmm system in {env}")
    assert env in ("waterbox", "vacuum", "complex")
    potential = MLPotential("ani2x")
    _, platform = check_implementation()

    ###################
    print(f"{platform=}")
    print(f"{env=}")
    ###################
    # TODO: add additional parameters for complex
    if env == "vacuum":
        mm_system = psf.createSystem(parameters, nonbondedMethod=NoCutoff)
    else:
        mm_system = psf.createSystem(parameters, nonbondedMethod=PME)

    print(f"{ml_atoms=}")

    #####################
    potential = MLPotential("ani2x")
    ml_system = potential.createMixedSystem(
        psf.topology, mm_system, ml_atoms, interpolate=True
    )
    #####################

    integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)
    platform = mm.Platform.getPlatformByName(platform)

    return Simulation(psf.topology, ml_system, integrator, platform=platform)


def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


def get_energy(sim):
    """get energy of system in a state"""
    return sim.context.getState(getEnergy=True).getPotentialEnergy()
