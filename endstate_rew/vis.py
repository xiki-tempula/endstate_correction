import pickle
import nglview as ng
from endstate_rew.system import generate_molecule
from endstate_rew.constant import zinc_systems
import mdtraj as md


def visualize_mol(
    zinc_id: str,
    forcefield: str,
    endstate: str,
    run_id: str = "",
    w_dir: str = "/data/shared/projects/endstate_rew",
    switching: bool = False,
    switching_length: int = 5001,
):

    # get name
    name, _ = zinc_systems[zinc_id]
    # generate molecule object
    m = generate_molecule(name=name, forcefield=forcefield)
    # write mol as pdb
    m.to_file("m.pdb", file_format="pdb")
    # get pickle file
    if not switching:
        # get correct file label
        if endstate == "mm":
            lamb_nr = "0.0000"
        elif endstate == "qml":
            lamb_nr = "1.0000"
        pickle_file = f"{w_dir}/{name}/sampling_{forcefield}/run{run_id}/{name}_samples_5000_steps_1000_lamb_{lamb_nr}.pickle"
    else:
        pickle_file = f"{w_dir}/{name}/switching_{forcefield}/{name}_samples_5000_steps_1000_lamb_{endstate}_endstate_nr_samples_500_switching_length_{switching_length}.pickle"
    # load traj
    f = pickle.load(open(pickle_file, "rb"))
    # load topology from pdb file
    # NOTE: pdb file is needed for mdtraj, which reads the topology
    # this is not very elegant # FIXME: try to load topology directly
    top = md.load("m.pdb").topology
    # generate trajectory instance
    traj = md.Trajectory(f, topology=top)
    # align traj
    traj.superpose(traj)
    view = ng.show_mdtraj(traj)
    return view
