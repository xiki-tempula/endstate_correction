import random

import numpy as np
from openmm import unit
from tqdm import tqdm
from typing import Tuple

from endstate_rew.constant import distance_unit, temperature, check_implementation
from endstate_rew.system import get_positions

from openmm import OpenMMException


def perform_switching(
    sim, lambdas: list, samples: list, nr_of_switches: int = 50, save_traj: bool = False
) -> Tuple[list, list]:
    """performs NEQ switching using the lambda sheme passed from randomly dranw samples"""

    implementation, platform = check_implementation()

    # list  of work values
    ws = []
    # list of conformations
    endstate_samples = []

    inst_switching = False
    if len(lambdas) == 2:
        print("Instantanious switching: dE will be calculated")
        inst_switching = True
    elif len(lambdas) < 2:
        raise RuntimeError("increase the number of lambda states")
    else:
        print("NEQ switching: dW will be calculated")

    # start with switch
    for _ in tqdm(range(nr_of_switches)):
        # select a random sample
        x = (
            np.array(random.choice(samples).value_in_unit(distance_unit))
            * distance_unit
        )
        # set position
        sim.context.setPositions(x)

        # reseed velocities
        sim.context.setVelocitiesToTemperature(temperature)

        # initialize work
        w = 0.0
        # perform NEQ switching
        for idx_lamb in range(1, len(lambdas)):
            # set lambda parameter
            sim.context.setParameter("lambda_interpolate", lambdas[idx_lamb])
            # test if neq or instantaneous swithching: if neq, perform integration step
            if not inst_switching:
                # perform 1 simulation step
                sim.step(1)
            # calculate work
            # evaluate u_t(x_t) - u_{t-1}(x_t)
            # calculate u_t(x_t)
            u_now = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # calculate u_{t-1}(x_t)
            sim.context.setParameter("lambda_interpolate", lambdas[idx_lamb - 1])
            u_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # add to accumulated work
            w += (u_now - u_before).value_in_unit(unit.kilojoule_per_mole)
        ws.append(w)
        if save_traj:
            endstate_samples.append(get_positions(sim))
    return np.array(ws) * unit.kilojoule_per_mole, endstate_samples
