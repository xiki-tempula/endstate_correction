# general imports
import pickle
import torch

# Imports from the openff toolkit
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

# import from openmm and ecosystem
from openmmml import MLPotential
import openmm as mm
from openmm.app import Simulation
from openmm import unit

forcefield = ForceField('openff_unconstrained-2.0.0.offxml')

from tqdm import tqdm

### define units
distance_unit = unit.angstrom
time_unit = unit.femtoseconds
speed_unit = distance_unit / time_unit

# constants
stepsize = 1 * time_unit
collision_rate = 1 / unit.picosecond
temperature = 300 * unit.kelvin

platform = 'cuda'
num_threads = 4
torch.set_num_threads(num_threads)

# generate a molecule using openff
###################
name = 'acetylacetone'
###################
molecule = Molecule.from_smiles('CC(C(C)=O)C(C)=O', hydrogens_are_explicit=False)
molecule.generate_conformers()

def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)

def collect_samples(sim, n_samples=1_000, n_steps_per_sample=10_000, lamb:float=0.0):
    """generate samples using a classical FF"""
    sim.context.setParameter('lambda', lamb)

    print(f'Generate samples with mixed System: {lamb=}, {n_samples=}, {n_steps_per_sample=}')   
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


potential = MLPotential('ani2x')
# generate a molecule using openff
system, topology = create_mm_system(molecule)
# define integrator
integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)
# define the atoms that are calculated using both potentials
ml_atoms = [atom.topology_atom_index for atom in topology.topology_atoms]
ml_system = potential.createMixedSystem(topology.to_openmm(), system, ml_atoms, interpolate=True)    

platform = mm.Platform.getPlatformByName('Reference')

sim = Simulation(topology, ml_system, integrator, platform=platform)
sim.context.setPositions(molecule.conformers[0])

mm_samples = collect_samples(sim, n_samples=1_000, n_steps_per_sample=5_000, lamb=0.0)
qml_samples = collect_samples(sim, n_samples=1_000, n_steps_per_sample=5_000, lamb=1.0)
pickle.dump(mm_samples, open(f'../data/{name}_mm_samples.pickle', 'wb+'))
pickle.dump(qml_samples, open('../data/{name}_qml_samples.pickle', 'wb+'))
