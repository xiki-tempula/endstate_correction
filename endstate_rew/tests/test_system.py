from openmm import unit
import numpy as np


def test_generate_molecule():
    from endstate_rew.system import generate_molecule

    # check that we can generate molecule and assert smiles before and after
    ethane_smiles = "CC"
    m = generate_molecule(ethane_smiles)
    assert ethane_smiles == m.to_smiles(explicit_hydrogens=False)

    # we require deterministic conformations, here we check that
    # coordinate set 0 is always the same
    coordinates = [
        [-7.45523175e-01, 4.14444551e-02, 1.17060656e-02],
        [7.47339732e-01, 2.87860116e-03, 1.22331042e-03],
        [-1.12970717e00, -6.37431527e-01, 8.14421395e-01],
        [-1.18489973e00, 1.02557024e00, 1.99636470e-01],
        [-1.19987084e00, -3.34603249e-01, -9.38879499e-01],
        [1.08415333e00, -7.36520284e-01, -7.73193261e-01],
        [1.22661485e00, 9.61738002e-01, -2.68072775e-01],
        [1.20189300e00, -3.23076234e-01, 9.53158295e-01],
    ]

    assert len(m.conformers) == 1
    compare_coordinates_to = m.conformers[0].value_in_unit(unit.angstrom)
    assert np.allclose(compare_coordinates_to, coordinates)


def test_sampling():
    from endstate_rew.system import (
        collect_samples,
        generate_molecule,
        initialize_simulation,
    )

    # sampling for ethanol
    # initialize molecule
    smiles = "CCO"
    molecule = generate_molecule(smiles)
    # initialize simulation and start sampling at MM endstate
    sim = initialize_simulation(molecule, at_endstate="MM", platform="CPU")
    mm_samples = collect_samples(sim, n_samples=5, n_steps_per_sample=10)
    # initialize simulation and start sampling at QML endstate
    sim = initialize_simulation(molecule, at_endstate="QML", platform="CPU")
    qml_samples = collect_samples(sim, n_samples=5, n_steps_per_sample=10)


def test_generate_simulation_instance():
    from endstate_rew.system import get_energy, generate_molecule, initialize_simulation

    # generate molecule
    ethane_smiles = "CC"
    m = generate_molecule(ethane_smiles)
    # initialize simulation for all thre cases
    _ = initialize_simulation(m, at_endstate="mm")
    _ = initialize_simulation(m, at_endstate="qml")
    _ = initialize_simulation(m)

    # check that potential that interpolats
    # returns the same values for the endstates
    # than the pure endstate implementation

    # at lambda=0.0 (mm endpoint)
    sim = initialize_simulation(m, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation(m)
    sim.context.setParameter("lambda", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_endstate, e_sim_mm_interpolate_endstate)

    # at lambda=1.0 (qml endpoint)
    sim = initialize_simulation(m, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation(m)
    sim.context.setParameter("lambda", 1.0)
    e_sim_qml_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )

    assert np.isclose(e_sim_qml_endstate, e_sim_qml_interpolate_endstate)

    # double check that QML and MM endpoint have different energies
    sim = initialize_simulation(m, at_endstate="mm")
    e_sim_mm_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    sim = initialize_simulation(m, at_endstate="qml")
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)

    assert not np.isclose(e_sim_mm_endstate, e_sim_qml_endstate)

def test_charmm_system_generation():
    from endstate_rew.system import get_charmm_system
    
  # list of all the charmm systems with the zinc id
    zinc_systems = [ 
    'ZINC00079729', 
    'ZINC00086442', 
    'ZINC00087557', 
    'ZINC00095858', 
    'ZINC00107550', 
    'ZINC00107778',
    'ZINC00123162', 
    'ZINC00133435', 
    'ZINC00138607', 
    'ZINC00140610', 
    'ZINC00164361', 
    'ZINC00167648', 
    'ZINC00169358', 
    'ZINC01036618', 
    'ZINC01755198', 
    'ZINC01867000', 
    'ZINC03127671', 
    'ZINC04344392', 
    'ZINC04363792', 
    'ZINC06568023', 
    'ZINC33381936']
    
    for zinc_id in zinc_systems:
       get_charmm_system(zinc_id, base = 'data/hipen_data')
