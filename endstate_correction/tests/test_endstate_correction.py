"""
Unit and regression test for the endstate_correction package.
"""

# Import package, test suite, and other packages as needed
import sys, pickle, os
import pytest


def test_endstate_correction_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "endstate_correction" in sys.modules


def test_FEP_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol
    from .test_neq import load_endstate_system_and_samples

    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name="ZINC00079729"
    )

    ####################################################
    # ----------------------- FEP ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="FEP",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0

    fep_protocol = Protocol(
        sim=sim,
        method="FEP",
        direction="bidirectional",
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == fep_protocol.nr_of_switches
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
def test_NEQ_protocol():
    """Perform NEQ uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol
    from .test_neq import load_endstate_system_and_samples

    system_name = "ZINC00079729"
    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="NEQ",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
        neq_switching_length=100,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == 0

    fep_protocol = Protocol(
        method="NEQ",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
        neq_switching_length=100,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == fep_protocol.nr_of_switches

    # generate data for plotting tests
    fep_protocol = Protocol(
        method="NEQ",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=100,
    )

    r = perform_endstate_correction(fep_protocol)
    pickle.dump(
        r,
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_bid.pickle", "wb"
        ),
    )

    fep_protocol = Protocol(
        method="NEQ",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=100,
    )

    r = perform_endstate_correction(fep_protocol)
    pickle.dump(
        r,
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_unid.pickle", "wb"
        ),
    )


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
def test_ALL_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import Protocol, perform_endstate_correction
    from .test_neq import load_endstate_system_and_samples
    import pickle

    system_name = "ZINC00079729"
    # start with NEQ
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ---------------- All corrections -----------------
    ####################################################

    protocol = Protocol(
        method="All",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=100,
    )

    r = perform_endstate_correction(protocol)
    pickle.dump(
        r,
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_all_corrections.pickle",
            "wb",
        ),
    )


def test_each_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol
    from .test_neq import load_endstate_system_and_samples

    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name="ZINC00079729"
    )

    ####################################################
    # ----------------------- FEP ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="FEP",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0

    fep_protocol = Protocol(
        sim=sim,
        method="FEP",
        direction="bidirectional",
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == fep_protocol.nr_of_switches
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    neq_protocol = Protocol(
        method="NEQ",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(neq_protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == neq_protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == 0

    fep_protocol = Protocol(
        sim=sim,
        method="NEQ",
        direction="bidirectional",
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == neq_protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == neq_protocol.nr_of_switches
