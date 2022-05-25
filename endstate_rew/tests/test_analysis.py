import numpy as np
import torch

### set number of CPU threads used by pytorch
num_threads = 2
torch.set_num_threads(num_threads)


def test_collect_equ_samples():
    from endstate_rew.analysis import _collect_equ_samples

    lambs = np.linspace(0, 1, 11)
    name = "ZINC00079729"
    path = "/data/shared/projects/endstate_rew/ZINC00079729/sampling_openff/run01/"
    samples = _collect_equ_samples(path, name="ZINC00079729", lambda_scheme=lambs)
    assert len(samples) + 5_000 == 11 * 5_000


def test_equilibrium_free_energy():
    from endstate_rew.analysis import calculate_u_ln

    smiles = "S=c1cc(-c2ccc(Cl)cc2)ss1"
    path = "/data/shared/projects/endstate_rew/ZINC00079729/sampling_openff/run01/"
    name = "ZINC00079729"

    calculate_u_ln(smiles, path, name)
