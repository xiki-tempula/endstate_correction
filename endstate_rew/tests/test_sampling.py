def test_sampling_with_openff():
    from endstate_rew.system import (
        generate_samples,
        generate_molecule,
        initialize_simulation_with_openff,
    )

    # generate molecule
    ethane_smiles = "CC"
    m = generate_molecule(forcefield="openff", smiles=ethane_smiles)

    # initialize simulation for all three cases
    sim = initialize_simulation_with_openff(m, at_endstate="mm")
    generate_samples(sim, 1, 50)
    sim = initialize_simulation_with_openff(m, at_endstate="qml")
    generate_samples(sim, 1, 50)
    sim = initialize_simulation_with_openff(m)
    generate_samples(sim, 1, 50)


def test_sampling_with_charmmff():
    from endstate_rew.system import (
        generate_samples,
        generate_molecule,
        initialize_simulation_with_charmmff,
        # remap_atoms,
    )

    # get zinc_id
    zinc_id = "ZINC00077329"
    smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    molecule = generate_molecule(forcefield="charmmff", smiles=smiles)

    from random import randint

    conf_id = randint(0, molecule.n_conformers - 1)
    print(f"select conf_id: {conf_id}")

    # initialize simulation for all thre cases
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, at_endstate="mm", conf_id=conf_id
    )
    generate_samples(sim, 1, 1_000)
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, at_endstate="qml", conf_id=conf_id
    )
    generate_samples(sim, 1, 1_000)
    sim = initialize_simulation_with_charmmff(molecule, zinc_id, conf_id=conf_id)
    generate_samples(sim, 1, 100)
