import pickle
import nglview as ng
from endstate_rew.system import generate_molecule
import mdtraj as md


def visualize_mol(smiles, pickle_file):

    m = generate_molecule(smiles)
    m.to_file('m.pdb', file_format='pdb')
    
    f = pickle.load(open(pickle_file, 'rb'))
    top=md.load('m.pdb').topology
    traj = md.Trajectory(f, topology=top)
    traj.superpose(traj)    
    view = ng.show_mdtraj(traj)
    return view