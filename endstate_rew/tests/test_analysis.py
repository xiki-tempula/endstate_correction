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
    path = "/home/mwieder/endstate_rew/data/ZINC00079729/sampling_openff/run01/"
    samples, N_k = _collect_equ_samples(path, name="ZINC00079729", lambda_scheme=lambs)
    print(N_k)
    assert N_k[0] == 2000
    assert len(samples) == 22000


def test_equilibrium_free_energy():
    from endstate_rew.analysis import calculate_u_kn
    from pymbar import MBAR

    smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    path = "/home/mwieder/endstate_rew/data/ZINC00079729/sampling_openff/run01/"
    name = "ZINC00079729"

    N_k, u_kn = calculate_u_kn(smiles, path, name, every_nth_frame=100, reload=False)

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -2105810.5891775307)

    N_k, u_kn = calculate_u_kn(smiles, path, name, every_nth_frame=100, reload=True)

    mbar = MBAR(u_kn, N_k)
    f = mbar.getFreeEnergyDifferences()
    assert np.isclose(mbar.f_k[-1], f[0][0][-1])
    assert np.isclose(f[0][0][-1], -2105810.5891775307)
