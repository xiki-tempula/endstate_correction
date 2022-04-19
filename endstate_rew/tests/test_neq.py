import pickle
from typing import Tuple

import numpy as np
from endstate_rew.neq import perform_switching
from endstate_rew.system import (
    _get_masses,
    _seed_velocities,
    create_mm_system,
    generate_molecule,
    initialize_simulation,
)
from openmm import unit
from openmm.app import Simulation


def load_system_and_samples(name: str, smiles: str) -> Tuple[Simulation, list, list]:
    # initialize simulation and load pre-generated samples

    n_samples = 2_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    molecule = generate_molecule(smiles)
    sim = initialize_simulation(molecule)

    samples_mm = pickle.load(
        open(
            f"data/{name}/sampling/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle",
            "rb",
        )
    )
    samples_qml = pickle.load(
        open(
            f"data/{name}/sampling/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle",
            "rb",
        )
    )

    return sim, samples_mm, samples_qml


def test_mass_list():

    molecule = generate_molecule(smiles="ClCCOCCCl")
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
    molecule = generate_molecule(smiles="ClCCOCCCl")
    system, _ = create_mm_system(molecule)
    _seed_velocities(_get_masses(system))


def test_switching():

    # load simulation and samples for 2cle
    sim, samples_mm, samples_qml = load_system_and_samples(
        name="2cle", smiles="ClCCOCCCl"
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
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), -3167768.70831208
    )
    lambs = np.linspace(1, 0, 2)
    print(lambs)
    dE_list = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), 3167768.70831208
    )

    # perform NEQ switching
    lambs = np.linspace(0, 1, 21)
    dW_forw = perform_switching(
        sim, lambdas=lambs, samples=samples_mm, nr_of_switches=1
    )
