# general imports
from endstate_correction.system import create_charmm_system, read_box
from openmm.app import (
    PME,
    CharmmParameterSet,
    CharmmPsfFile,
    PDBFile,
)
from endstate_correction.analysis import plot_endstate_correction_results
import endstate_correction
from endstate_correction.protocol import perform_endstate_correction, Protocol
import mdtraj
from openmm import unit

########################################################
########################################################
# ------------ set up the waterbox system --------------
# we use a system that is shipped with the repo
package_path = endstate_correction.__path__[0]
system_name = "1_octanol"
env = "waterbox"
# define the output directory
output_base = f"{system_name}/"
parameter_base = f"{package_path}/data/jctc_data"
# load the charmm specific files (psf, pdb, rtf, prm and str files)
psf_file = f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.psf"
psf = CharmmPsfFile(psf_file)
pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.pdb")
params = CharmmParameterSet(
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.rtf",
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.prm",
    f"{parameter_base}/toppar/top_all36_cgenff.rtf",
    f"{parameter_base}/toppar/par_all36_cgenff.prm",
    f"{parameter_base}/toppar/toppar_water_ions.str",
)
# set up the treatment of the system for the specific environment
if env == "waterbox":
    psf = read_box(psf, f"{parameter_base}/{system_name}/charmm-gui/input.config.dat")

# define region that should be treated with the qml
chains = list(psf.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
# define system
sim = create_charmm_system(psf=psf, parameters=params, env=env, ml_atoms=ml_atoms)

########################################################
########################################################
# ------------------- load samples ---------------------#
n_samples = 5_000
n_steps_per_sample = 1_000
traj_base = f"{system_name}/"
mm_samples = []
traj = mdtraj.load_dcd(
    f"{traj_base}/equilibrium_samples/run01/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000_{env}.dcd",
    top=psf_file,
)

if env == "waterbox":
    traj.image_molecules()
    
mm_samples.extend(traj.xyz * unit.nanometer)  # NOTE: this is in nanometer!
###############################
qml_samples = []
traj = mdtraj.load_dcd(
    f"{traj_base}/equilibrium_samples/run01/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000_{env}.dcd",
    top=psf_file,
)

if env == "waterbox":
    traj.image_molecules()

qml_samples.extend(traj.xyz * unit.nanometer)  # NOTE: this is in nanometer!

####################################################
# ----------------------- FEP ----------------------
####################################################

fep_protocol = Protocol(
    method="NEQ",
    direction="bidirectional",
    sim=sim,
    trajectories=[mm_samples, qml_samples],
    nr_of_switches=50,
    neq_switching_length=100,
)

r = perform_endstate_correction(fep_protocol)
plot_endstate_correction_results(system_name, r, "results_neq_bidirectional.png")
