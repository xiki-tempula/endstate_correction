# general imports
from endstate_correction.constant import (
    temperature,
)
from endstate_correction.system import create_charmm_system, read_box
import numpy as np
from openmm.app import (
    PME,
    CharmmParameterSet,
    CharmmPsfFile,
    PDBFile,
    DCDReporter,
)
from endstate_correction.equ import generate_samples
import endstate_correction

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
psf = CharmmPsfFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.psf")
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

##############################################################
# ------------------ Start equilibrium sampling ---------------
# define equilibirum sampling control parameters
run_id = 1
n_samples = 5_000
n_steps_per_sample = 1_000
# path where samples should be stored
base = f"{output_base}/equilibrium_samples/run{run_id:0>2d}"
# define lambda states
lambs = np.linspace(0, 1, 11)
for lamb in lambs:
    print(f"{lamb=}")
    trajectory_file = f"{base}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}_{env}.dcd"

    print(f"Trajectory saved to: {trajectory_file}")
    # set lambda
    sim.context.setParameter("lambda_interpolate", lamb)
    # set coordinates
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature)
    # collect samples
    sim.reporters.append(
        DCDReporter(
            trajectory_file,
            n_steps_per_sample,
        )
    )

    samples = generate_samples(
        sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
    )
    sim.reporters.clear()
