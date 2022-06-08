import numpy as np
import torch

### set number of CPU threads used by pytorch
num_threads = 2
torch.set_num_threads(num_threads)


def test_collect_equ_samples():
    """test if we are able to collect samples as anticipated"""
    from endstate_rew.analysis import _collect_equ_samples

    lambs = np.linspace(0, 1, 11)
    name = "ZINC00079729"
    path = "data/ZINC00079729/sampling_openff/run01/"
    samples, N_k = _collect_equ_samples(path, name="ZINC00079729", lambda_scheme=lambs)
    print(N_k)
    assert N_k[0] == 2000
    assert len(samples) == 22000

    samples, N_k = _collect_equ_samples(
        path, name="ZINC00079729", lambda_scheme=lambs, only_endstates=True
    )
    print(N_k)
    assert N_k[0] == 2000
    assert N_k[-1] == 2000
    assert len(samples) == 4000


def test_collect_neq_samples():
    """test if we are able to collect samples as anticipated"""
    from endstate_rew.analysis import _collect_neq_samples

    nr_of_switches = 200
    paths = [
        f"data/ZINC00077329/switching_charmmff/ZINC00077329_neq_ws_from_mm_to_qml_{nr_of_switches}_5001.pickle"
    ]
    ws = _collect_neq_samples(paths)
    assert len(ws) == nr_of_switches


def test_equilibrium_free_energy_charmmff():
    "test that u_kn can be calculated and that results are consistent whether we reload mbar pickle or regernerate it"
    from endstate_rew.analysis import calculate_u_kn
    from pymbar import MBAR

    smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    name = "ZINC00079729"
    path = f"data/{name}/sampling_charmmff/run01/"

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield="charmmff",
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=False,
        override=True,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -2105818.3584215776, rtol=1e-06)

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield="charmmff",
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=True,
        override=False,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -2105818.3584215776, rtol=1e-06)


def test_equilibrium_free_energy_openff():
    "test that u_kn can be calculated and that results are consistent whether we reload mbar pickle or regernerate it"
    from endstate_rew.analysis import calculate_u_kn
    from pymbar import MBAR

    smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    path = "data/ZINC00079729/sampling_openff/run01/"
    name = "ZINC00079729"

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield="openff",
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=False,
        override=True,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -2105810.5, rtol=1e-06)

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield="openff",
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=True,
    )

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -2105810.5, rtol=1e-06)


def test_plotting_equilibrium_free_energy():
    "Test that plotting functions can be called"
    from endstate_rew.analysis import calculate_u_kn
    from endstate_rew.analysis import (
        plot_overlap_for_equilibrium_free_energy,
        plot_results_for_equilibrium_free_energy,
    )

    smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    path = "data/ZINC00079729/sampling_openff/run01/"
    name = "ZINC00079729"

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield="openff",
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=False,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=name)

    path = "data/ZINC00079729/sampling_charmmff/run01/"

    N_k, u_kn = calculate_u_kn(
        smiles=smiles,
        forcefield="charmmff",
        path_to_files=path,
        name=name,
        every_nth_frame=100,
        reload=False,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=name)
