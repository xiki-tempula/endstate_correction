"""
Unit and regression test for the endstate_correction package.
"""

# Import package, test suite, and other packages as needed
import sys


def test_endstate_correction_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "endstate_correction" in sys.modules  


def test_FEP_protocoll():
    """Perform FEP uni- and bidirectional protocoll"""
    from endstate_correction.protocoll import perform_endstate_correction, Protocoll
    from .test_neq import load_endstate_system_and_samples

    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name="ZINC00079729"
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
    assert len(r.dE_mm_to_qml) == fep_protocoll.nr_of_switches
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0

    fep_protocoll = Protocoll(
        sim=sim,
        method="FEP",
        direction="bidirectional",
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocoll)
    assert len(r.dE_mm_to_qml) == fep_protocoll.nr_of_switches
    assert len(r.dE_qml_to_mm) == fep_protocoll.nr_of_switches
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0


def test_NEQ_protocoll():
    """Perform NEQ uni- and bidirectional protocoll"""
    from endstate_correction.protocoll import perform_endstate_correction, Protocoll
    from .test_neq import load_endstate_system_and_samples

    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name="ZINC00079729"
    )

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    fep_protocoll = Protocoll(
        method="NEQ",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
        neq_switching_length=100,
    )

    r = perform_endstate_correction(fep_protocoll)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == fep_protocoll.nr_of_switches
    assert len(r.W_qml_to_mm) == 0

    fep_protocoll = Protocoll(
        method="NEQ",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
        neq_switching_length=100,
    )

    r = perform_endstate_correction(fep_protocoll)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == fep_protocoll.nr_of_switches
    assert len(r.W_qml_to_mm) == fep_protocoll.nr_of_switches


def test_EQU_protocoll():
    """Perform equilibrium free energy protocoll"""
    from endstate_correction.protocoll import perform_endstate_correction, Protocoll
    from .test_equ import load_equ_samples
    from openmm.app import CharmmParameterSet, CharmmPsfFile
    import pathlib
    import endstate_correction
    from endstate_correction.system import create_charmm_system

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    path = pathlib.Path(endstate_correction.__file__).resolve().parent

    hipen_testsystem = f"{path}/data/hipen_data"
    system_name = "ZINC00077329"
    system_name = "ZINC00077329"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )

    sim = create_charmm_system(psf=psf, parameters=params, env="vacuum", tlc="UNK")

    # load all equ samples
    trajs = load_equ_samples(system_name=system_name)

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    fep_protocoll = Protocoll(
        method="EQU", sim=sim, trajectories=trajs, equ_every_nth_frame=50
    )

    r = perform_endstate_correction(fep_protocoll)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0
    # test that mbar instance was created
    r.equ_mbar.getFreeEnergyDifferences()
