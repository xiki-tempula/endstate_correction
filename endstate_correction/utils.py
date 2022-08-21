import pickle
import mdtraj as md
from openmm import unit
from openmm.app import CharmmPsfFile, CharmmCrdFile, PDBFile

#def convert_pickle_to_

def convert_pickle_to_dcd_file(
    pickle_file_path: str,
    path_to_topology: str,
    path_to_coords: str,
    dcd_output_path: str,
    pdb_output_path: str,
):
    # helper function that converts pickle trajectory file to dcd file
    
    f = pickle.load(open(pickle_file_path, "rb"))
    traj = [frame.value_in_unit(unit.nanometer) for frame in f]
    topology = CharmmPsfFile(path_to_topology).topology
    positions = CharmmCrdFile(path_to_coords)

    PDBFile.writeFile(topology, positions.positions, file=open(pdb_output_path, "w"))
    traj = md.Trajectory(traj, topology=topology)
    traj.save_dcd(dcd_output_path)
    print("Finished converting ...")
