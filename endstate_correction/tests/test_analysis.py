import glob
import pathlib

import endstate_correction
import mdtraj as md
import numpy as np
from endstate_correction.analysis import calculate_u_kn
from endstate_correction.system import create_charmm_system
from openmm import unit
from openmm.app import CharmmParameterSet, CharmmPsfFile
from pymbar import MBAR


def test_collect_equ_samples():
    """test if we are able to collect samples as anticipated"""
    from endstate_correction.analysis import _collect_equ_samples

    name = "ZINC00077329"
    nr_of_samples = 5_000
    nr_of_steps = 1_000
    trajs = []
    path = f"data/{name}/sampling_charmmff/run01/"
    for lamb in np.linspace(0, 1, 11):
        print(lamb)
        file = glob.glob(
            f"{path}/{name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.dcd"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        if len(file) == 0:
            print("WARNING! Incomplete equ sampling. Proceed with cautions.")

        trajs.append(md.open(file[0]).read()[0] * unit.nanometer)

    samples, N_k = _collect_equ_samples(trajs)

    print(N_k)
    assert N_k[0] == 400
    assert len(samples) == 4400
    trajs = []
    for lamb in np.linspace(0, 1, 2):
        print(lamb)
        file = glob.glob(
            f"{path}/{name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.dcd"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        if len(file) == 0:
            print("WARNING! Incomplete equ sampling. Proceed with cautions.")

        trajs.append(md.open(file[0]).read()[0] * unit.nanometer)

    samples, N_k = _collect_equ_samples(trajs)
    print(N_k)
    assert N_k[0] == 400
    assert N_k[-1] == 400
    assert len(samples) == 800

    mm_samples = samples[: int(N_k[0])]
    qml_samples = samples[int(N_k[0]) :]
    assert len(mm_samples) == 400
    assert len(qml_samples) == 400


def test_collect_work_values():
    """test if we are able to collect samples as anticipated"""
    from endstate_correction.analysis import _collect_work_values

    nr_of_switches = 200
    path = f"data/ZINC00077329/switching_charmmff/ZINC00077329_neq_ws_from_mm_to_qml_{nr_of_switches}_5001.pickle"
    ws = _collect_work_values(path)
    assert len(ws) == nr_of_switches


def test_equilibrium_free_energy():
    "test that u_kn can be calculated and that results are consistent whether we reload mbar pickle or regernerate it"
    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    path = pathlib.Path(endstate_correction.__file__).resolve().parent
    hipen_testsystem = f"{path}/data/hipen_data"

    system_name = "ZINC00077329"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )

    sim = create_charmm_system(psf=psf, parameters=params, env="vacuum", tlc="UNK")

    nr_of_samples = 5_000
    nr_of_steps = 1_000
    trajs = []
    path = f"data/{system_name}/sampling_charmmff/run01/"

    for lamb in np.linspace(0, 1, 11):
        print(lamb)
        file = glob.glob(
            f"{path}/{system_name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.dcd"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        if len(file) == 0:
            print("WARNING! Incomplete equ sampling. Proceed with cautions.")

        trajs.append(md.open(file[0]).read()[0] * unit.angstrom)

    N_k, u_kn = calculate_u_kn(
        path_to_files=path,
        sim=sim,
        trajs=trajs,
        every_nth_frame=20,
        reload=False,
        override=True,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -940544.0390218807, rtol=1e-06)

    N_k, u_kn = calculate_u_kn(
        path_to_files=path,
        trajs=trajs,
        every_nth_frame=20,
        sim=sim,
        reload=True,
        override=False,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -940544.0390218807, rtol=1e-06)


def test_plotting_equilibrium_free_energy():
    "Test that plotting functions can be called"
    from endstate_correction.analysis import (
        calculate_u_kn,
        plot_overlap_for_equilibrium_free_energy,
        plot_results_for_equilibrium_free_energy,
    )

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    path = pathlib.Path(endstate_correction.__file__).resolve().parent
    hipen_testsystem = f"{path}/data/hipen_data"

    system_name = "ZINC00077329"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )

    sim = create_charmm_system(psf=psf, parameters=params, env="vacuum", tlc="UNK")

    nr_of_samples = 5_000
    nr_of_steps = 1_000
    trajs = []
    path = f"data/{system_name}/sampling_charmmff/run01/"

    for lamb in np.linspace(0, 1, 11):
        print(lamb)
        file = glob.glob(
            f"{path}/{system_name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.dcd"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        if len(file) == 0:
            print("WARNING! Incomplete equ sampling. Proceed with cautions.")

        trajs.append(md.open(file[0]).read()[0] * unit.angstrom)

    N_k, u_kn = calculate_u_kn(
        path_to_files=path,
        trajs=trajs,
        every_nth_frame=20,
        sim=sim,
        reload=False,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)
