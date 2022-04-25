import pickle
import nglview as ng
from endstate_rew.system import generate_molecule
import mdtraj as md


def visualize_mol(smiles: str, pickle_file: str):
    """Takes a smiles string and the path to a trajectory pickle file and returns
    a nglview instance to visualize the MD simulation.

    Args:
        smiles (str): smiles sting defining the molecule
        pickle_file (str): file path to pickled trajectorie

    Returns:
        _type_: nglview view instance
    """

    m = generate_molecule(smiles)
    m.to_file("m.pdb", file_format="pdb")

    f = pickle.load(open(pickle_file, "rb"))
    top = md.load("m.pdb").topology
    traj = md.Trajectory(f, topology=top)
    traj.superpose(traj)
    view = ng.show_mdtraj(traj)
    return view
