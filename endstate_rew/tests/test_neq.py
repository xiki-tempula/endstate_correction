import pickle
from typing import Tuple

import numpy as np
from endstate_rew.neq import perform_switching
from endstate_rew.system import (
    _get_masses,
    _seed_velocities,
    create_mm_system,
    generate_molecule,
    initialize_simulation_with_openff,
    create_charmm_system,
    initialize_simulation_with_charmmff,
)
from openmm import unit
from openmm.app import Simulation


def load_endstate_system_and_samples_charmmff(
    molecule, name: str, path_to_samples: str, base: str = "data/hipen_data"
) -> Tuple[Simulation, list, list]:
    # initialize simulation and load pre-generated samples

    n_samples = 5_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    sim = initialize_simulation_with_charmmff(molecule, name, base)

    samples_mm = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.pickle",
            "rb",
        )
    )
    samples_qml = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.pickle",
            "rb",
        )
    )

    return sim, samples_mm, samples_qml


def load_endstate_system_and_samples_openff(
    name: str,
    smiles: str,
    path_to_samples: str,
) -> Tuple[Simulation, list, list]:
    # initialize simulation and load pre-generated samples

    n_samples = 5_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    molecule = generate_molecule(forcefield="openff", smiles=smiles)
    sim = initialize_simulation_with_openff(molecule)

    samples_mm = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.pickle",
            "rb",
        )
    )
    samples_qml = pickle.load(
        open(
            f"{path_to_samples}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.pickle",
            "rb",
        )
    )

    return sim, samples_mm, samples_qml


def test_mass_list():

    molecule = generate_molecule(forcefield="openff", smiles="ClCCOCCCl")
    system, _ = create_mm_system(molecule)

    # make sure that mass list generated from system and molecuel are the same
    m_list = _get_masses(system)
    masses = np.array([a.mass / unit.dalton for a in molecule.atoms]) * unit.daltons
    print(m_list)
    print(masses)
    assert np.allclose(
        m_list.value_in_unit(unit.daltons), masses.value_in_unit(unit.daltons)
    )


def test_seed_velocities():

    # test that manual velocity seeding works
    # openff
    molecule = generate_molecule(forcefield="openff", smiles="ClCCOCCCl")
    system, _ = create_mm_system(molecule)
    _seed_velocities(_get_masses(system))

    # charmmff
    molecule = generate_molecule(
        forcefield="charmmff", name="ZINC00079729", base="data/hipen_data"
    )
    system, _ = create_mm_system(molecule)
    _seed_velocities(_get_masses(system))


def test_switching_openff():

    # load simulation and samples for 2cle
    sim, samples_mm, samples_qml = load_endstate_system_and_samples_openff(
        name="ZINC00079729",
        smiles="S=c1cc(-c2ccc(Cl)cc2)ss1",
        path_to_samples="data/ZINC00079729/sampling_openff/run01",
    )
    # perform instantaneous switching with predetermined coordinate set
    # here, we evaluate dU_forw = dU(x)_qml - dU(x)_mm and make sure that it is the same as
    # dU_rev = dU(x)_mm - dU(x)_qml
    lambs = np.linspace(0, 1, 2)
    print(lambs)
    dE_list = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), -5252603.00305137
    )
    lambs = np.linspace(1, 0, 2)
    print(lambs)
    dE_list = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), 5252603.00305137
    )

    # perform NEQ switching
    lambs = np.linspace(0, 1, 21)
    dW_forw = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dW_forw)
    assert np.isclose(dW_forw.value_in_unit(unit.kilojoule_per_mole), -5252599.97640173)

    # perform NEQ switching
    lambs = np.linspace(0, 1, 101)
    dW_forw = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dW_forw)
    assert np.isclose(dW_forw.value_in_unit(unit.kilojoule_per_mole), -5252596.88091529)


def test_switching_charmmff():

    name, smiles = "ZINC00079729", "S=c1cc(-c2ccc(Cl)cc2)ss1"
    molecule = generate_molecule(
        forcefield="charmmff",
        smiles=smiles,
        base="data/hipen_data",
    )

    # load simulation and samples for ZINC00077329
    sim, samples_mm, samples_qml = load_endstate_system_and_samples_charmmff(
        molecule=molecule,
        name=name,
        base="data/hipen_data",
        path_to_samples="data/ZINC00079729/sampling_openff/run01",
    )
    # perform instantaneous switching with predetermined coordinate set
    # here, we evaluate dU_forw = dU(x)_qml - dU(x)_mm and make sure that it is the same as
    # dU_rev = dU(x)_mm - dU(x)_qml
    lambs = np.linspace(0, 1, 2)
    print(lambs)
    dE_list = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), -6787758.792709583
    )
    lambs = np.linspace(1, 0, 2)
    print(lambs)
    dE_list = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), 6787758.792709583
    )

    # perform NEQ switching
    lambs = np.linspace(0, 1, 21)
    dW_forw = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )

    print(dW_forw)

    # perform NEQ switching
    lambs = np.linspace(0, 1, 101)
    dW_forw = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )

    print(dW_forw)
