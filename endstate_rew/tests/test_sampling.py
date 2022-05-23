def test_sampling_with_openff():
    from endstate_rew.system import (
        collect_samples,
        generate_molecule,
        initialize_simulation_with_openff,
    )

    # generate molecule
    # ethane_smiles = "CC"
    zinc_id = "ZINC00077329"
    # m = generate_molecule(ethane_smiles)
    m = generate_molecule(name=zinc_id, forcefield="openff")
    # initialize simulation for all three cases
    sim = initialize_simulation_with_openff(m, at_endstate="mm")
    collect_samples(sim, 1, 50)
    sim = initialize_simulation_with_openff(m, at_endstate="qml")
    collect_samples(sim, 1, 50)
    sim = initialize_simulation_with_openff(m)
    collect_samples(sim, 1, 50)


def test_sampling_with_charmmff():
    from endstate_rew.system import (
        collect_samples,
        generate_molecule,
        initialize_simulation_with_charmmff,
        # remap_atoms,
    )

    # get zinc_id
    zinc_id = "ZINC00077329"
    # smiles = "Cn1cc(Cl)c(/C=N/O)n1"
    # molecule = generate_molecule(smiles)
    molecule = generate_molecule(
        name=zinc_id, forcefield="charmmff", base="data/hipen_data"
    )
    # molecule = remap_atoms(zinc_id, base="data/hipen_data", molecule=molecule)

    from random import randint

    conf_id = randint(0, molecule.n_conformers - 1)
    print(f"select conf_id: {conf_id}")

    # initialize simulation for all thre cases
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="mm", conf_id=conf_id
    )
    collect_samples(sim, 1, 2_000)
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="qml", conf_id=conf_id
    )
    collect_samples(sim, 1, 2_000)
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", conf_id=conf_id
    )
    collect_samples(sim, 1, 2000)
