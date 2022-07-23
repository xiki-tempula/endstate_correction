import numpy as np
from openmm import unit
import endstate_correction
import pathlib
from openmm.app import (
    CharmmParameterSet,
    CharmmPsfFile,
    PDBFile,
    CharmmCrdFile,
)

path = pathlib.Path(endstate_correction.__file__).resolve().parent
hipen_testsystem = f"{path}/data/hipen_data"

path = pathlib.Path(endstate_correction.__file__).resolve().parent
jctc_testsystem = f"{path}/data/jctc_data"


def test_generate_simulation_instances_with_charmmff():
    from endstate_correction.system import create_charmm_system, get_energy, read_box

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files

    system_name = "ZINC00079729"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    crd = CharmmCrdFile(f"{hipen_testsystem}/{system_name}/{system_name}.crd")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )

    sim = create_charmm_system(psf=psf, parameters=params, env="vacuum", tlc="UNK")
    sim.context.setPositions(crd.positions)

    ############################
    ############################
    # check potential at endpoints
    # at lambda=0.0 (mm endpoint)
    sim.context.setParameter("lambda_interpolate", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_interpolate_endstate, 156.9957913623657)

    ############################
    ############################
    # at lambda=1.0 (qml endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_qml_endstate)
    assert np.isclose(e_sim_qml_endstate, -5252411.066221259)

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    system_name = "1_octanol"
    psf = CharmmPsfFile(f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/vac.psf")
    pdb = PDBFile(f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/vac.pdb")
    params = CharmmParameterSet(
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.rtf",
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.prm",
        f"{jctc_testsystem}/toppar/top_all36_cgenff.rtf",
        f"{jctc_testsystem}/toppar/par_all36_cgenff.prm",
        f"{jctc_testsystem}/toppar/toppar_water_ions.str",
    )

    sim = create_charmm_system(psf=psf, parameters=params, env="vacuum", tlc="UNK")
    sim.context.setPositions(pdb.positions)

    ############################
    ############################
    # check potential at endpoints
    # at lambda=0.0 (mm endpoint)
    sim.context.setParameter("lambda_interpolate", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_interpolate_endstate, 316.4088125228882)

    ############################
    ############################
    # at lambda=1.0 (qml endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_qml_endstate)
    assert np.isclose(e_sim_qml_endstate, -1025774.735780582)

    ########################################################
    ########################################################
    # ----------------- waterbox ---------------------------
    # get all relevant files
    system_name = "1_octanol"
    psf = CharmmPsfFile(
        f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/step3_input.psf"
    )
    pdb = PDBFile(f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/step3_input.pdb")
    params = CharmmParameterSet(
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.rtf",
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.prm",
        f"{jctc_testsystem}/toppar/top_all36_cgenff.rtf",
        f"{jctc_testsystem}/toppar/par_all36_cgenff.prm",
        f"{jctc_testsystem}/toppar/toppar_water_ions.str",
    )
    psf = read_box(psf, f"{jctc_testsystem}/{system_name}/charmm-gui/input.config.dat")

    sim = create_charmm_system(psf=psf, parameters=params, env="waterbox", tlc="UNK")
    sim.context.setPositions(pdb.positions)

    ############################
    ############################
    # check potential at endpoints
    # at lambda=0.0 (mm endpoint)
    sim.context.setParameter("lambda_interpolate", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_interpolate_endstate, -41853.389923448354)

    ############################
    ############################
    # at lambda=1.0 (qml endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_qml_endstate)
    assert np.isclose(e_sim_qml_endstate, -1067965.9293421314)
