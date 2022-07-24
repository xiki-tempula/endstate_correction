import pathlib

import endstate_correction
from endstate_correction.system import create_charmm_system
from openmm import unit
from openmm.app import CharmmParameterSet, CharmmPsfFile


def test_plotting_equilibrium_free_energy():
    "Test that plotting functions can be called"
    from endstate_correction.analysis import (
        plot_overlap_for_equilibrium_free_energy,
        plot_results_for_equilibrium_free_energy,
    )
    from endstate_correction.equ import calculate_u_kn
    from .test_equ import load_equ_samples

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
        every_nth_frame=50,
        sim=sim,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)


def test_FEP_protocoll():
    """Perform FEP uni- and bidirectional protocoll"""
    from endstate_correction.protocoll import perform_endstate_correction, Protocoll
    from .test_neq import load_endstate_system_and_samples
    from endstate_correction.analysis import plot_endstate_correction_results

    system_name = "ZINC00079729"
    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ----------------------- FEP ----------------------
    ####################################################

    fep_protocoll = Protocoll(
        method="FEP",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocoll)
    plot_endstate_correction_results(system_name, r, "results_fep_unidirectional.png")

    fep_protocoll = Protocoll(
        method="FEP",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocoll)
    plot_endstate_correction_results(system_name, r, "results_fep_bidirectional.png")
