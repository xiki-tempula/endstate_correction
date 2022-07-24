import glob
import pathlib
import pickle

import endstate_correction
import mdtraj as md
import numpy as np
from endstate_correction.equ import calculate_u_kn
from endstate_correction.system import create_charmm_system
from openmm import unit
from openmm.app import CharmmParameterSet, CharmmPsfFile


def test_collect_work_values():
    """test if we are able to collect samples as anticipated"""
    from endstate_correction.analysis import _collect_work_values

    nr_of_switches = 200
    path = f"data/ZINC00077329/switching_charmmff/ZINC00077329_neq_ws_from_mm_to_qml_{nr_of_switches}_5001.pickle"
    ws = _collect_work_values(path)
    assert len(ws) == nr_of_switches


def test_plotting_equilibrium_free_energy():
    "Test that plotting functions can be called"
    from endstate_correction.analysis import (
        plot_overlap_for_equilibrium_free_energy,
        plot_results_for_equilibrium_free_energy,
    )
    from endstate_correction.equ import calculate_u_kn, _collect_equ_samples
    """test if we are able to plot overlap and """
    
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
    trajs = load_equ_samples(system_name)


    N_k, u_kn = calculate_u_kn(
        trajs=trajs,
        every_nth_frame=20,
        sim=sim,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)
