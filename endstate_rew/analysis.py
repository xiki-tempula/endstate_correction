import glob
import pickle
from collections import namedtuple
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openmm import unit
from pymbar import BAR, EXP

from endstate_rew.constant import kBT


def _collect_samples(path:str, name:str, direction:str = 'mm_to_qml') ->list:
        
    files = glob.glob(f"{path}/{name}_*_{direction}_{name}_*.pickle")
    ws = []
    for f in files:
        w_ = pickle.load(open(f, 'rb')).value_in_unit(unit.kilojoule_per_mole)
        ws.extend(w_)
    number_of_samples = len(ws)
    print(f"Number of samples used: {number_of_samples}")
    return ws * unit.kilojoule_per_mole

def collect_results(name:str, smiles: str)->NamedTuple:
    from endstate_rew.neq import perform_switching
    from endstate_rew.system import initialize_simulation, generate_molecule

    w_dir = f'../data/{name}/switching/run1'
    # load samples
    mm_samples = pickle.load(open(f'../data/{name}/sampling/{name}_mm_samples_2000_1000.pickle', 'rb'))
    qml_samples = pickle.load(open(f'../data/{name}/sampling/{name}_qml_samples_2000_1000.pickle', 'rb'))
    
    # get pregenerated work values
    ws_from_mm_to_qml = np.array(_collect_samples(w_dir, name, 'mm_to_qml')/ kBT)
    ws_from_qml_to_mm = np.array(_collect_samples(w_dir, name,'qml_to_mm')/ kBT)
    
    
    # perform instantenious swichting (FEP) to get dE values
    switching_length = 2
    nr_of_switches = 500
    # create molecule
    molecule = generate_molecule(smiles)
    sim = initialize_simulation(molecule)
    lambs = np.linspace(0,1,switching_length)
    dEs_from_mm_to_qml = np.array(
        perform_switching(sim, lambs, samples=mm_samples,nr_of_switches=nr_of_switches)/kBT)
    lambs = np.linspace(1,0,switching_length)
    dEs_from_qml_to_mm = np.array(
        perform_switching(sim,lambs, samples=qml_samples,nr_of_switches=nr_of_switches)/kBT)

    # pack everything in a namedtuple
    Results = namedtuple('Results', 'dWs_from_mm_to_qml dWs_from_qml_to_mm dEs_from_mm_to_qml dEs_from_qml_to_mm')
    results = Results(ws_from_mm_to_qml, ws_from_qml_to_mm, dEs_from_mm_to_qml, dEs_from_qml_to_mm)
    return results


def plot_resutls_of_switching_experiments(name:str, results:NamedTuple):
    
    
    print('################################')
    print(f"Crooks' equation: {BAR(results.dWs_from_mm_to_qml, results.dWs_from_qml_to_mm)}")
    print(f"Jarzynski's equation: {EXP(results.dWs_from_mm_to_qml)}")
    print(f"Zwanzig's equation: {EXP(results.dEs_from_mm_to_qml)}")
    print(f"Zwanzig's equation bidirectional: {BAR(results.dEs_from_mm_to_qml, results.dEs_from_qml_to_mm)}")
    print('################################')
    
    
    sns.set_context("talk")
    fig, axs = plt.subplots(3,1, figsize=(11.0, 9), dpi=600)
    # plot distribution of dE and dW
    #########################################
    axs[0].set_title(fr'{name} - distribution of $\Delta$W and $\Delta$E')
    palett = sns.color_palette(n_colors=8)
    palett_as_hex = palett.as_hex()
    c1, c2, c3, c4 = palett_as_hex[0], palett_as_hex[1], palett_as_hex[2], palett_as_hex[3]
    axs[0].ticklabel_format(axis='x', style='sci', useOffset=True ,scilimits=(0,0))
    #axs[1].ticklabel_format(axis='x', style='sci', useOffset=False,scilimits=(0,0))

    sns.histplot(ax=axs[0], alpha=0.5, data=results.dWs_from_mm_to_qml*-1, kde=True, stat='density', label=r'$\Delta$W(MM$\rightarrow$QML)', color=c1)
    sns.histplot(ax=axs[0], alpha=0.5, data=results.dEs_from_mm_to_qml*-1, kde=True, stat='density', label=r'$\Delta$E(MM$\rightarrow$QML)', color=c2)
    sns.histplot(ax=axs[0], alpha=0.5, data=results.dWs_from_qml_to_mm, kde=True, stat='density', label=r'$\Delta$W(QML$\rightarrow$MM)', color=c3)
    sns.histplot(ax=axs[0], alpha=0.5, data=results.dEs_from_qml_to_mm, kde=True, stat='density', label=r'$\Delta$E(QML$\rightarrow$MM)', color=c4)
    axs[0].legend()

    # plot results
    #########################################
    axs[1].set_title(fr'{name} - offset $\Delta$G(MM$\rightarrow$QML)')
    #Crooks' equation
    ddG_list, dddG_list = [],[] 
    ddG, dddG = BAR(results.dWs_from_mm_to_qml, results.dWs_from_qml_to_mm)
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # Jarzynski's equation
    ddG, dddG = EXP(results.dWs_from_mm_to_qml)
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # FEP
    ddG, dddG = EXP(results.dEs_from_mm_to_qml)
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # FEP + BAR
    ddG, dddG = BAR(results.dEs_from_mm_to_qml, results.dEs_from_qml_to_mm)
    ddG_list.append(ddG)
    dddG_list.append(dddG)

    axs[1].errorbar([i for i in range(len(ddG_list))], ddG_list-np.min(ddG_list),  dddG_list,  fmt='o')
    axs[1].set_xticklabels(["", 'Crooks', "", 'Jazynski', "", 'FEP+EXP', "", 'FEP+BAR'])
    axs[1].set_ylabel('kT')
    axs[1].legend()

    # plot cummulative stddev of dE and dW
    #########################################
    axs[2].set_title(fr'{name} - cummulative stddev of $\Delta$W and $\Delta$E')

    cum_stddev_ws_from_mm_to_qml = [results.dWs_from_mm_to_qml[:x].std() for x in range(1,len(results.dWs_from_mm_to_qml)+1)]
    cum_stddev_ws_from_qml_to_mm = [results.dWs_from_qml_to_mm[:x].std() for x in range(1,len(results.dWs_from_qml_to_mm)+1)]

    cum_stddev_dEs_from_mm_to_qml = [results.dEs_from_mm_to_qml[:x].std() for x in range(1,len(results.dEs_from_mm_to_qml)+1)]
    cum_stddev_dEs_from_qml_to_mm = [results.dEs_from_qml_to_mm[:x].std() for x in range(1,len(results.dEs_from_qml_to_mm)+1)]
    axs[2].plot(cum_stddev_ws_from_mm_to_qml, label=r'stddev $\Delta$W(MM$\rightarrow$QML)')
    axs[2].plot(cum_stddev_ws_from_qml_to_mm , label=r'stddev $\Delta$W(QML$\rightarrow$MM)')
    axs[2].plot(cum_stddev_dEs_from_mm_to_qml, label=r'stddev $\Delta$E(MM$\rightarrow$QML)')
    axs[2].plot(cum_stddev_dEs_from_qml_to_mm , label=r'stddev $\Delta$E(QML$\rightarrow$MM)')
    # plot 1 kT limit
    axs[2].axhline(y = 1, color = 'r', linestyle = '-')

    axs[2].set_ylabel('kT')

    axs[2].legend()

    plt.tight_layout()
    plt.show()
