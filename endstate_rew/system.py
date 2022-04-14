import numpy as np
import openmm as mm
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit
from openmm.app import Simulation
from openmmml import MLPotential
from tqdm import tqdm
from openmm.app import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet
from openmm.app import NoCutoff
from openmm.unit import unit

from endstate_rew.constant import collision_rate, stepsize, temperature, kBT, speed_unit

forcefield = ForceField('openff_unconstrained-2.0.0.offxml')

def generate_molecule(smiles:str)->Molecule:
    # generate a molecule using openff
    molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
    molecule.generate_conformers()
    return molecule

def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)

def get_energy(sim):
    """get energy of system in a state"""
    return sim.context.getState(getEnergy=True).getPotentialEnergy()


def collect_samples(sim, n_samples:int=1_000, n_steps_per_sample:int=10_000):
    """generate samples using a defined system"""

    print(f'Generate samples with mixed System: {n_samples=}, {n_steps_per_sample=}')   
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

def _seed_velocities(sim: Simulation, molecule: Molecule)->np.ndarray:
    
    # should only take
    # sim.context.setVelocitiesToTemperature(temperature)
    # but currently this returns a pytorch error
    # instead seed manually from boltzmann distribution
    
    # generate mass arrays
    masses = np.array([a.mass/unit.dalton for a in molecule.atoms]) * unit.daltons
    sigma_v = (
        np.array([unit.sqrt(kBT / m) / speed_unit for m in masses])
        * speed_unit
    )

    return np.random.randn(len(sigma_v), 3) * sigma_v[:, None]
    


def initialize_simulation(molecule:Molecule, at_endstate:str='', platform:str='CPU'):
    """Initialize a simulation instance

    Args:
        molecule (Molecule): _description_
        at_endstate (str, optional): _description_. Defaults to ''.
        platform (str, optional): _description_. Defaults to 'CPU'.

    Returns:
        _type_: _description_
    """
    assert molecule.n_conformers > 0
    
    # initialize potential
    potential = MLPotential('ani2x')
    # generate a molecule using openff
    system, topology = create_mm_system(molecule)
    # define integrator
    integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)
    platform = mm.Platform.getPlatformByName(platform)

    # define the atoms that are calculated using both potentials
    if not at_endstate:
        ml_atoms = [atom.topology_atom_index for atom in topology.topology_atoms]
        ml_system = potential.createMixedSystem(topology.to_openmm(), system, ml_atoms, interpolate=True)    
        sim = Simulation(topology, ml_system, integrator, platform=platform)
    elif at_endstate.upper() == 'QML':
        system = potential.createSystem(topology.to_openmm())
        sim = Simulation(topology, system, integrator, platform=platform)
        print('Initializing QML system')
    elif at_endstate.upper() == 'MM':
        sim = Simulation(topology, system, integrator, platform=platform)
        print('Initializing MM system')
    
    sim.context.setPositions(molecule.conformers[0])
    #NOTE: minimizing the energy of the interpolating potential leeds to very high energies,
    # for now avoiding call to minimizer    
    #sim.minimizeEnergy(maxIterations=100)
    
    #NOTE: velocities are seeded manually right now (otherwise pytorch error) -- 
    # this will be fiexed in the future 
    #FIXME: revert back to openMM velovity call 
    #sim.context.setVelocitiesToTemperature(temperature)
    sim.context.setVelocities(_seed_velocities(sim, molecule))
    return sim

#creating charmm systems from zinc data
def get_charmm_system(name:str):
    
    #define base of the path
    base = '../data/hipen_data/'
    
    # get psf, crd and prm files
    psf = CharmmPsfFile(f'{base}/{name}/{name}.psf')
    crd = CharmmCrdFile(f'{base}/{name}/{name}.crd')
    params = CharmmParameterSet(f'{base}/top_all36_cgenff.rtf', f'{base}/par_all36_cgenff.prm', f'{base}/{name}/{name}.str')
    
    # define system object
    system = psf.createSystem(params, nonbondedMethod=NoCutoff)
    
    # return system object
    return system