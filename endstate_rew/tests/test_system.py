import numpy as np
from openmm import unit


def test_conf_selection():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule  # , remap_atoms

    for ff in ["openff", "charmmff"]:
        for zinc_name, _ in zinc_systems:
            print(zinc_name)
            if (
                zinc_name == "ZINC00061095"
                or zinc_name == "ZINC00095858"
                or zinc_name == "ZINC00138607"
            ):  # skip system that has wrong topology
                continue

            # m = generate_molecule(smiles)
            m = generate_molecule(name=zinc_name, forcefield=ff, base="data/hipen_data")
            assert len(m.conformers) >= 1

        """ for zinc_name, smiles in zinc_systems:
        print(zinc_name)
        if (
            zinc_name == "ZINC00061095"
            or zinc_name == "ZINC00095858"
            or zinc_name == "ZINC00138607"
        ):  # skip system that has wrong topology
            continue

        print(zinc_name)
        m = generate_molecule(smiles)
        m = remap_atoms(zinc_name, base="data/hipen_data", molecule=m) 

        assert len(m.conformers) >= 1 """


""" def test_confs():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule, remap_atoms

    for zinc_name, smiles in zinc_systems:
        print(zinc_name)
        if (
            zinc_name == "ZINC00061095"
            or zinc_name == "ZINC00095858"
            or zinc_name == "ZINC00138607"
        ):  # skip system that has wrong topology
            continue

        m = generate_molecule(smiles)
        m = remap_atoms(zinc_name, base="data/hipen_data", molecule=m)

        compare_coordinates_to = m.conformers[0].value_in_unit(unit.angstrom) """


def test_generate_molecule():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule

    # check that we can generate molecule and assert smiles before and after (openff)
    # ethane_smiles = "CC"
    zinc_id = zinc_systems[1][0]
    smiles = zinc_systems[1][1]
    # m = generate_molecule(ethane_smiles)
    m_openff = generate_molecule(name=zinc_id, forcefield="openff")
    # assert ethane_smiles == m.to_smiles(explicit_hydrogens=False)
    assert smiles == m_openff.to_smiles(explicit_hydrogens=False)

    # make sure that atom indexing of charmm molecule is the same as in the psf file
    m_charmmff = generate_molecule(
        name=zinc_id, forcefield="charmmff", base="data/hipen_data"
    )

    # we require deterministic conformations, here we check that
    # coordinate set 0 is always the same
    """ coordinates = [
        [-7.45523175e-01, 4.14444551e-02, 1.17060656e-02],
        [7.47339732e-01, 2.87860116e-03, 1.22331042e-03],
        [-1.12970717e00, -6.37431527e-01, 8.14421395e-01],
        [-1.18489973e00, 1.02557024e00, 1.99636470e-01],
        [-1.19987084e00, -3.34603249e-01, -9.38879499e-01],
        [1.08415333e00, -7.36520284e-01, -7.73193261e-01],
        [1.22661485e00, 9.61738002e-01, -2.68072775e-01],
        [1.20189300e00, -3.23076234e-01, 9.53158295e-01],
    ] """

    coordinates = [
        [-2.75987995, -0.69053967, 0.36029375],
        [-1.47374083, -0.07361633, 0.09984744],
        [-1.33655965, 1.18256186, -0.37147202],
        [0.01006323, 1.38777693, -0.47988788],
        [0.80189666, 2.86426872, -1.04883321],
        [0.66095509, 0.24515119, -0.06958264],
        [2.09923885, 0.03630211, -0.02586892],
        [2.65278589, -1.05267619, 0.36663243],
        [4.01325859, -1.25211235, 0.40867711],
        [-0.311077, -0.63920528, 0.28300144],
        [-3.50614014, -0.46794238, -0.43942482],
        [-2.63891528, -1.78669473, 0.37180877],
        [-3.17350775, -0.27589788, 1.29720489],
        [-2.09406379, 1.89931201, -0.62057146],
        [2.7391263, 0.83811731, -0.33966023],
        [4.31655976, -2.21480533, 0.20783536],
    ]

    # for mol in [m_openff, m_charmmff]:
    assert len(m_openff.conformers) == 1
    compare_coordinates_to = m_openff.conformers[0].value_in_unit(unit.angstrom)
    assert np.allclose(compare_coordinates_to, coordinates)


def test_atom_indices():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import generate_molecule

    # collect atom indices of psf file
    base = "data/hipen_data"
    name = zinc_systems[1][0]
    mol = generate_molecule(name=name, forcefield="charmmff", base="data/hipen_data")
    n_atoms = mol.n_atoms

    # get psf file
    file = open(f"{base}/{name}/{name}.psf")
    lines = file.readlines()

    atoms_psf = []
    line_count = 0
    atom_count = 0

    for line in lines:
        # atoms begin in line 8
        if line_count >= 7:
            atom_count += 1
            tmp = line.split()
            atoms_psf.append(tmp[4])
        # break when all atom indices are collected
        if atom_count == n_atoms:
            break
        line_count += 1

    # collect atom indices of pdb file
    # save a pdb file of the generated molecule
    mol.to_file("tmp.pdb", "pdb")
    file_pdb = open(f"tmp.pdb")

    lines = file_pdb.readlines()
    atoms_pdb = []
    line_count = 0
    atom_count = 0

    for line in lines:
        atom_count += 1
        tmp = line.split()
        atoms_pdb.append(tmp[2])
        if atom_count == n_atoms:
            break
        line_count += 1

    # make sure, that atom indexing after molecule generation is the same as in the psf file
    assert atoms_psf == atoms_pdb


def test_sampling():
    from endstate_rew.system import (
        collect_samples,
        generate_molecule,
        initialize_simulation_with_charmmff,
        initialize_simulation_with_openff,
    )
    from endstate_rew.constant import zinc_systems

    # sample with openff
    # sampling for ethanol
    # initialize molecule
    # smiles = "CCO"
    # molecule = generate_molecule(smiles)
    zinc_id = zinc_systems[1][0]
    molecule = generate_molecule(name=zinc_id, forcefield="openff")
    # initialize simulation and start sampling at MM endstate
    sim = initialize_simulation_with_openff(molecule, at_endstate="MM", platform="CPU")
    mm_samples = collect_samples(sim, n_samples=5, n_steps_per_sample=10)
    # initialize simulation and start sampling at QML endstate
    sim = initialize_simulation_with_openff(molecule, at_endstate="QML", platform="CPU")
    qml_samples = collect_samples(sim, n_samples=5, n_steps_per_sample=10)

    # sample with charmmff
    # generate zinc mol
    # zinc_id = "ZINC00079729"
    # smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    # molecule = generate_molecule(smiles)
    molecule = generate_molecule(
        name=zinc_id, forcefield="openff", base="data/hipen_data"
    )
    # initialize simulation for all thre cases
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="mm"
    )
    mm_samples = collect_samples(sim, n_samples=5, n_steps_per_sample=10)
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="qml"
    )
    qml_samples = collect_samples(sim, n_samples=5, n_steps_per_sample=10)


def test_generate_simulation_instances_with_openff():
    from endstate_rew.system import (
        generate_molecule,
        get_energy,
        initialize_simulation_with_openff,
    )
    from endstate_rew.constant import zinc_systems

    # generate molecule
    # ethane_smiles = "CC"
    zinc_id = zinc_systems[1][0]
    # m = generate_molecule(ethane_smiles)
    m = generate_molecule(name=zinc_id, forcefield="openff")
    # initialize simulation for all three cases
    _ = initialize_simulation_with_openff(m, at_endstate="mm")
    _ = initialize_simulation_with_openff(m, at_endstate="qml")
    _ = initialize_simulation_with_openff(m)

    # check that potential that interpolats
    # returns the same values for the endstates
    # than the pure endstate implementation

    # at lambda=0.0 (mm endpoint)
    sim = initialize_simulation_with_openff(m, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_openff(m)
    sim.context.setParameter("lambda", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)

    # at lambda=1.0 (qml endpoint)
    sim = initialize_simulation_with_openff(m, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_openff(m)
    sim.context.setParameter("lambda", 1.0)
    e_sim_qml_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )

    assert np.isclose(e_sim_qml_endstate, e_sim_qml_interpolate_endstate)

    # double check that QML and MM endpoint have different energies
    sim = initialize_simulation_with_openff(m, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_openff(m, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    assert not np.isclose(e_sim_mm_endstate, e_sim_qml_endstate)


""" def test_atom_mapping_smiles_hipen():
    # test if we are able to remap the atom order
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import create_charmm_system, generate_molecule
    from openff.toolkit.topology import Molecule

    # list of all the charmm systems with the zinc id

    for zinc_name, smiles in zinc_systems:
        print(zinc_name)
        if (
            zinc_name == "ZINC00061095"
            or zinc_name == "ZINC00095858"
            or zinc_name == "ZINC00138607"
        ):  # skip system that has wrong topology
            continue

        # generate mol
        molecule = generate_molecule(smiles)

        # generate the charmm system
        system, topology, _ = create_charmm_system(zinc_name, base="data/hipen_data")

        # generate a mol with the atom order from the psf/crd files and the original
        # smiles to define bond order etc
        m_new = Molecule.from_pdb_and_smiles("tmp.pdb", molecule.to_smiles())

        print(molecule.to_smiles(mapped=True))
        print(m_new.to_smiles(mapped=True))

        # make sure original smiles is unequal new smiles
        assert molecule.to_smiles(mapped=True) != m_new.to_smiles(mapped=True) """


def test_charmm_system_generation():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import (
        create_charmm_system,
        generate_molecule,
        initialize_simulation_with_charmmff,
        # remap_atoms,
    )

    # list of all the charmm systems with the zinc id

    for zinc_name, smiles in zinc_systems:
        print(zinc_name)
        if (
            zinc_name == "ZINC00061095"
            or zinc_name == "ZINC00095858"
            or zinc_name == "ZINC00138607"
        ):  # skip system that has wrong topology
            continue
        # molecule = generate_molecule(smiles)
        # molecule = remap_atoms(zinc_name, base="data/hipen_data", molecule=molecule)
        molecule = generate_molecule(
            name=zinc_name, forcefield="charmmff", base="data/hipen_data"
        )

        create_charmm_system(zinc_name, base="data/hipen_data")
        _ = initialize_simulation_with_charmmff(
            molecule, zinc_name, base="data/hipen_data", at_endstate="mm"
        )


def test_generate_simulation_instances_with_charmmff():
    from endstate_rew.constant import zinc_systems
    from endstate_rew.system import (
        generate_molecule,
        get_energy,
        initialize_simulation_with_charmmff,
        # remap_atoms,
    )

    # get zinc_id

    # zinc_id = "ZINC00079729"
    # smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    # molecule = generate_molecule(smiles)
    # molecule = remap_atoms(zinc_id, base="data/hipen_data", molecule=molecule)
    zinc_id = zinc_systems[1][0]
    molecule = generate_molecule(
        name=zinc_id, forcefield="charmmff", base="data/hipen_data"
    )

    # initialize simulation for all thre cases
    _ = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="mm"
    )
    _ = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="qml"
    )
    _ = initialize_simulation_with_charmmff(molecule, zinc_id, base="data/hipen_data")

    # check that potential that interpolates
    # returns the same values for the endstates
    # than the pure endstate implementation

    # at lambda=0.0 (mm endpoint)
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="mm"
    )
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_charmmff(molecule, zinc_id, base="data/hipen_data")
    sim.context.setParameter("lambda", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)

    # at lambda=1.0 (qml endpoint)
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="qml"
    )
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_charmmff(molecule, zinc_id, base="data/hipen_data")
    sim.context.setParameter("lambda", 1.0)
    e_sim_qml_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )

    assert np.isclose(e_sim_qml_endstate, e_sim_qml_interpolate_endstate)

    # double check that QML and MM endpoint have different energies
    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="mm"
    )
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation_with_charmmff(
        molecule, zinc_id, base="data/hipen_data", at_endstate="qml"
    )
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    assert not np.isclose(e_sim_mm_endstate, e_sim_qml_endstate)
