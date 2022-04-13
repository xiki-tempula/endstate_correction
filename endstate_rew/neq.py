import random

import numpy as np
from openmm import unit
from tqdm import tqdm

from endstate_rew.constant import distance_unit


def perform_switching(sim, lambdas:list, samples:list, nr_of_switches:int=50)->list:
    """performs NEQ switching using the lambda sheme passed from randomly dranw samples"""
    
    # list  of work values
    ws = []
    # start with switch
    for _ in tqdm(range(nr_of_switches)):
        # select a random sample
        x = np.array(random.choice(samples).value_in_unit(distance_unit)) * distance_unit
        # initialize work
        w = 0.0
        # set position    
        sim.context.setPositions(x)
        
        # perform NEQ switching
        for idx_lamb in range(1,len(lambdas)):
            # set lambda parameter
            sim.context.setParameter('lambda', lambdas[idx_lamb])
            # perform 1 simulation step
            sim.step(1)
            # calculate work
            # evaluate u_t(x_t) - u_{t-1}(x_t)
            # calculate u_t(x_t)
            u_now = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # calculate u_{t-1}(x_t)
            sim.context.setParameter('lambda', lambdas[idx_lamb-1])
            u_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # add to accumulated work
            w += (u_now - u_before).value_in_unit(unit.kilojoule_per_mole)

        ws.append(w)
    return np.array(ws) * unit.kilojoule_per_mole

def perform_inst_switching(sim, lambdas:list, samples:list, nr_of_switches:int=50)->list:
    """performs NEQ switching using the lambda sheme passed from randomly dranw samples"""
    
    # list  of work values
    ws = []
    # start with switch
    for _ in tqdm(range(nr_of_switches)):
        # select a random sample
        x = np.array(random.choice(samples).value_in_unit(distance_unit)) * distance_unit
        # initialize work
        w = 0.0
        # set position    
        sim.context.setPositions(x)
        
        # perform NEQ switching
        for idx_lamb in range(1,len(lambdas)):
            # set lambda parameter
            sim.context.setParameter('lambda', lambdas[idx_lamb])
            # calculate work
            # evaluate u_t(x_t) - u_{t-1}(x_t)
            # calculate u_t(x_t)
            u_now = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # calculate u_{t-1}(x_t)
            sim.context.setParameter('lambda', lambdas[idx_lamb-1])
            u_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # add to accumulated work
            w += (u_now - u_before).value_in_unit(unit.kilojoule_per_mole)

        ws.append(w)
    return np.array(ws) * unit.kilojoule_per_mole