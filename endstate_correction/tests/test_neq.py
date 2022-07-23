import pickle
from typing import Tuple

import numpy as np
import pytest
from endstate_correction.neq import perform_switching
from endstate_correction.system import (
    _get_hipen_data,
    create_openff_system,
    generate_molecule,
    initialize_simulation_with_charmmff,
    initialize_simulation_with_openff,
)
from openmm import unit
from openmm.app import Simulation


def load_endstate_system_and_samples_charmmff(
    molecule, name: str, path_to_samples: str, base: str = ""
) -> Tuple[Simulation, list, list]:
    # initialize simulation and load pre-generated samples
    try:
        from NNPOps import OptimizedTorchANI as _

        implementation = "NNPOps"
        platform = "CUDA"
    except ModuleNotFoundError:
        platform = "CPU"

    if not base:
        base = _get_hipen_data()

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


@pytest.mark.parametrize(
    "ff, dW_for, dW_rev",
    [
        (
            "charmmff",
            -2405742.1451317305,
            2405742.1452882225,
        ),
        (
            "openff",
            -2346353.8752537486,
            2346353.8752537486,
        ),
    ],
)
def test_switching(ff, dW_for, dW_rev):

    name = "ZINC00077329"
    print(f"{name=}")
    print(f"{ff=}")
    # load simulation and samples for 2cle
    sim, samples_mm, samples_qml = load_endstate_system_and_samples_openff(
        name=name,
        smiles="Cn1cc(Cl)c(/C=N/O)n1",
        path_to_samples=f"data/{name}/sampling_{ff}/run01",
    )
    # perform instantaneous switching with predetermined coordinate set
    # here, we evaluate dU_forw = dU(x)_qml - dU(x)_mm and make sure that it is the same as
    # dU_rev = dU(x)_mm - dU(x)_qml
    lambs = np.linspace(0, 1, 2)
    print(lambs)
    dE_list, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(dE_list[0].value_in_unit(unit.kilojoule_per_mole), dW_for)
    lambs = np.linspace(1, 0, 2)
    print(lambs)
    dE_list, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(dE_list[0].value_in_unit(unit.kilojoule_per_mole), dW_rev)

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
