import pickle
from glob import glob
from os import path

import numpy as np
import openmm as mm
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit
from openmm.app import (
    CharmmCrdFile,
    CharmmParameterSet,
    CharmmPsfFile,
    NoCutoff,
    Simulation,
)
from openmmml import MLPotential
from torch import randint
from tqdm import tqdm

from endstate_rew.constant import collision_rate, kBT, speed_unit, stepsize, temperature

forcefield = ForceField("openff_unconstrained-2.0.0.offxml")


def generate_molecule(smiles: str, nr_of_conformations: int = 10) -> Molecule:
    # generate a molecule using openff
    molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
    molecule.generate_conformers(n_conformers=nr_of_conformations)
    molecule.max_conf = nr_of_conformations
    return molecule


def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


def get_energy(sim):
    """get energy of system in a state"""
    return sim.context.getState(getEnergy=True).getPotentialEnergy()


def collect_samples(sim, n_samples: int = 1_000, n_steps_per_sample: int = 10_000):
    """generate samples using a defined system"""

    print(f"Generate samples with mixed System: {n_samples=}, {n_steps_per_sample=}")
    samples = []
    for _ in tqdm(range(n_samples)):
        sim.step(n_steps_per_sample)
        samples.append(get_positions(sim))
    return samples


def create_mm_system(molecule):
    """given a molecule it creates an openMM system and topology instance"""
    topology = molecule.to_topology()
    system = forcefield.create_openmm_system(topology)
    return system, topology


def _get_masses(system) -> np.array:
    return (
        np.array(
            [
                system.getParticleMass(atom_idx) / unit.dalton
                for atom_idx in range(system.getNumParticles())
            ]
        )
        * unit.daltons
    )


def _seed_velocities(masses: np.array) -> np.ndarray:

    # should only take
    # sim.context.setVelocitiesToTemperature(temperature)
    # but currently this returns a pytorch error
    # instead seed manually from boltzmann distribution

    sigma_v = np.array([unit.sqrt(kBT / m) / speed_unit for m in masses]) * speed_unit

    return np.random.randn(len(sigma_v), 3) * sigma_v[:, None]


def _initialize_simulation(
    at_endstate: str, topology, potential, molecule, conf_id: int, platform, system
):
    # define integrator
    integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)
    platform = mm.Platform.getPlatformByName(platform)

    # define the atoms that are calculated using both potentials
    if not at_endstate:
        # ml_atoms = [atom.topology_atom_index for atom in topology.topology_atoms]
        ml_atoms = [atom.index for atom in topology.atoms()]
        ml_system = potential.createMixedSystem(
            topology, system, ml_atoms, interpolate=True
        )
        sim = Simulation(topology, ml_system, integrator, platform=platform)
    elif at_endstate.upper() == "QML":
        system = potential.createSystem(topology)
        sim = Simulation(topology, system, integrator, platform=platform)
        print("Initializing QML system")
    elif at_endstate.upper() == "MM":
        sim = Simulation(topology, system, integrator, platform=platform)
        print("Initializing MM system")

    sim.context.setPositions(molecule.conformers[conf_id])
    # NOTE: FIXME: minimizing the energy of the interpolating potential leeds to very high energies,
    # for now avoiding call to minimizer
    # sim.minimizeEnergy(maxIterations=100)

    # NOTE: FIXME: velocities are seeded manually right now (otherwise pytorch error) --
    # this will be fiexed in the future
    # revert back to openMM velovity call
    # sim.context.setVelocitiesToTemperature(temperature)
    sim.context.setVelocities(_seed_velocities(_get_masses(system)))
    return sim


def initialize_simulation_with_openff(
    molecule: Molecule,
    at_endstate: str = "",
    platform: str = "CPU",
    w_dir="",
    conf_id: int = 0,
):
    """Initialize a simulation instance

    Args:
        molecule (Molecule): _description_
        at_endstate (str, optional): _description_. Defaults to ''.
        platform (str, optional): _description_. Defaults to 'CPU'.

    Returns:
        _type_: _description_
    """

    assert molecule.n_conformers > 0
    print("Using openff ...")
    # initialize potential
    potential = MLPotential("ani2x")
    # initialize openMM system and topology
    if w_dir:
        mol_path = f"{w_dir}/system.openff"
        if path.isfile(mol_path):  # if already generated, load it
            print("load system ...")
            system, topology = pickle.load(open(mol_path, "rb"))
        else:  # if not generated, generate it and save it
            print("generate and save system ...")
            system, topology = create_mm_system(molecule)
            pickle.dump((system, topology), open(mol_path, "wb+"))
    else:
        system, topology = create_mm_system(molecule)

    return _initialize_simulation(
        at_endstate,
        topology.to_openmm(),
        potential,
        molecule,
        conf_id,
        platform,
        system,
    )


# creating charmm systems from zinc data
def create_charmm_system(name: str, base="../data/hipen_data"):
    from openmm.app import PDBFile

    # check if input directory exists
    if not path.isdir(base):
        raise RuntimeError("Path is not a directory.")

    # check if input directory contains at least one directory with the name 'ZINC'
    if len(glob(base + "/ZINC*")) < 1:
        raise RuntimeError("No ZINC directory found.")

    # get psf, crd and prm files
    psf = CharmmPsfFile(f"{base}/{name}/{name}.psf")
    crd = CharmmCrdFile(f"{base}/{name}/{name}.crd")
    params = CharmmParameterSet(
        f"{base}/top_all36_cgenff.rtf",
        f"{base}/par_all36_cgenff.prm",
        f"{base}/{name}/{name}.str",
    )

    # define system object
    system = psf.createSystem(params, nonbondedMethod=NoCutoff)
    PDBFile.writeFile(psf.topology, crd.positions, open("tmp.pdb", "w+"))
    # return system object
    return system, psf.topology, crd.positions


def remap_atoms(zinc_id: str, base: str, molecule):

    _, _, _ = create_charmm_system(zinc_id, base)
    new_m = Molecule.from_pdb_and_smiles("tmp.pdb", molecule.to_smiles())
    new_m.generate_conformers(n_conformers=molecule.max_conf)
    return new_m


# initialize simulation charmm system
def initialize_simulation_with_charmmff(
    molecule: Molecule,
    zinc_id: str,
    base: str = "../data/hipen_data",
    at_endstate: str = "",
    platform: str = "CPU",
    conf_id: int = 0,
):
    """Initialize a simulation instance

    Args:
        zinc_id (str): _description_
        base (str, optional): _description_. Defaults to '../data/hipen_data'
        at_endstate (str, optional): _description_. Defaults to ''.
        platform (str, optional): _description_. Defaults to 'CPU'.

    Returns:
        _type_: _description_
    """
    from openff.toolkit.topology import Molecule

    assert molecule.n_conformers > 0

    # initialize potential
    potential = MLPotential("ani2x")
    # generate the charmm system
    system, topology, _ = create_charmm_system(zinc_id, base)

    return _initialize_simulation(
        at_endstate, topology, potential, molecule, conf_id, platform, system
    )
