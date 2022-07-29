import pathlib

import endstate_correction
from endstate_correction.system import create_charmm_system
from openmm.app import CharmmParameterSet, CharmmPsfFile
import pytest
import os


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
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
    # define region that should be treated with the qml
    chains = list(psf.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]

    sim = create_charmm_system(
        psf=psf, parameters=params, env="vacuum", ml_atoms=ml_atoms
    )
    trajs = load_equ_samples(system_name)

    N_k, u_kn = calculate_u_kn(
        trajs=trajs,
        every_nth_frame=50,
        sim=sim,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)


def test_plot_results_for_FEP_protocoll():
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
        nr_of_switches=100,
    )

    r = perform_endstate_correction(fep_protocoll)
    plot_endstate_correction_results(system_name, r, "results_fep_bidirectional.png")


def test_plot_results_for_NEQ_protocoll():
    """Perform FEP uni- and bidirectional protocoll"""
    from endstate_correction.protocoll import perform_endstate_correction, Protocoll
    from .test_neq import load_endstate_system_and_samples
    from endstate_correction.analysis import plot_endstate_correction_results
    import pickle

    system_name = "ZINC00079729"
    # start with NEQ
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    fep_protocoll = Protocoll(
        method="NEQ",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=100,
    )

    # r = perform_endstate_correction(fep_protocoll)
    # pickle.dump(r, open(f"neq_unid.pickle", "wb"))

    r = pickle.load(
        open(f"data/{system_name}/switching_charmmff/neq_unid.pickle", "rb")
    )
    plot_endstate_correction_results(
        system_name, r, "{system_name}_results_neq_unidirectional.png"
    )

    fep_protocoll = Protocoll(
        method="NEQ",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=100,
    )

    # r = perform_endstate_correction(fep_protocoll)
    # pickle.dump(r, open(f"neq_bid.pickle", "wb"))

    r = pickle.load(open(f"data/{system_name}/switching_charmmff/neq_bid.pickle", "rb"))
    plot_endstate_correction_results(
        system_name, r, "{system_name}_results_neq_bidirectional.png"
    )
