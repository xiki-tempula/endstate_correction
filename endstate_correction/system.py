# general imports
import json

import openmm as mm
from tqdm import tqdm
from openmm import unit
from openmm.app import PME, CharmmParameterSet, CharmmPsfFile, NoCutoff, Simulation
from openmmml import MLPotential

from endstate_correction.constant import collision_rate, stepsize, temperature


def read_box(psf, filename: str):
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
    tlc: str,
):

    ###################
    print(f"Generating charmm system in {env}")
    assert env in ("waterbox", "vacuum", "complex")
    potential = MLPotential("ani2x")
    ff = "charmmff"
    platform = "CUDA"
    ###################
    print(f"{ff=}")
    print(f"{platform=}")
    print(f"{env=}")
    ###################
    # TODO: add additional parameters for complex
    if env == "vacuum":
        mm_system = psf.createSystem(parameters, nonbondedMethod=NoCutoff)
    else:
        mm_system = psf.createSystem(parameters, nonbondedMethod=PME)

    # TODO: check lingand automatically
    chains = list(psf.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
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


def generate_samples(sim, n_samples: int = 1_000, n_steps_per_sample: int = 10_000):
    """generate samples using a defined system"""

    print(f"Generate samples with mixed System: {n_samples=}, {n_steps_per_sample=}")
    samples = []
    for _ in tqdm(range(n_samples)):
        sim.step(n_steps_per_sample)
        samples.append(get_positions(sim))
    return samples
