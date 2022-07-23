import pickle
from typing import Tuple

import numpy as np
from endstate_correction.neq import perform_switching
from openmm import unit
from openmm.app import Simulation
import endstate_correction
import pathlib
import mdtraj


def load_endstate_system_and_samples(
    system_name: str,
) -> Tuple[Simulation, list, list]:

    # initialize simulation and load pre-generated samples

    from openmm.app import (
        CharmmParameterSet,
        CharmmPsfFile,
        CharmmCrdFile,
    )

    from endstate_correction.system import create_charmm_system

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    path = pathlib.Path(endstate_correction.__file__).resolve().parent
    hipen_testsystem = f"{path}/data/hipen_data"

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
    # data/ZINC00079729/sampling_charmmff/run01/ZINC00079729_samples_5000_steps_1000_lamb_0.0000.pickle
    n_samples = 5_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    mm_samples = []
    traj = mdtraj.load_dcd(
        f"data/{system_name}/sampling_charmmff/run01/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.dcd",
        top=f"{hipen_testsystem}/{system_name}/{system_name}.psf",
    )

    mm_samples.extend(traj.xyz * unit.nanometer)
    qml_samples = []
    traj = mdtraj.load_dcd(
        f"data/{system_name}/sampling_charmmff/run01/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.dcd",
        top=f"{hipen_testsystem}/{system_name}/{system_name}.psf",
    )
    qml_samples.extend(traj.xyz * unit.nanometer)

    return sim, mm_samples, qml_samples


def test_switching():

    system_name = "ZINC00077329"
    print(f"{system_name=}")

    # load simulation and samples for 2cle
    sim, samples_mm, samples_qml = load_endstate_system_and_samples(
        system_name=system_name,
    )
    # perform instantaneous switching with predetermined coordinate set
    # here, we evaluate dU_forw = dU(x)_qml - dU(x)_mm and make sure that it is the same as
    # dU_rev = dU(x)_mm - dU(x)_qml
    lambs = np.linspace(0, 1, 2)
    print(lambs)
    dE_list, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), -5252555.14554809
    )
    lambs = np.linspace(1, 0, 2)

    dE_list, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dE_list)

    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), 5252555.14554809
    )

    # perform NEQ switching
    lambs = np.linspace(0, 1, 21)
    dW_forw, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dW_forw)

    # perform NEQ switching
    lambs = np.linspace(0, 1, 101)
    dW_forw, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dW_forw)

    # check return values
    lambs = np.linspace(0, 1, 2)
    list_1, list_2 = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1, save_traj=False
    )
    assert len(list_1) != 0 and len(list_2) == 0

    list_1, list_2 = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1, save_traj=True
    )
    assert len(list_1) != 0 and len(list_2) != 0
