from dataclasses import dataclass
from openmm.app import Simulation
import numpy as np
from dataclasses import dataclass, field
from typing import List

from pymbar import MBAR


@dataclass
class Protocoll:
    method: str
    sim: Simulation
    trajectories: List
    direction: str = "None"
    nr_of_switches: int = -1
    neq_switching_length: int = 5_000
    equ_nr_of_lambda_states: int = 11
    equ_every_nth_frame: int = 10


@dataclass
class Results:
    dE_mm_to_qml: List = field(default_factory=lambda: [])
    dE_qml_to_mm: List = field(default_factory=lambda: [])
    W_mm_to_qml: List = field(default_factory=lambda: [])
    W_qml_to_mm: List = field(default_factory=lambda: [])
    equ_mbar: MBAR = None


def perform_endstate_correction(protocoll: Protocoll):

    from endstate_correction.neq import perform_switching
    from endstate_correction.constant import kBT

    # check that all necessary keywords are present
    if protocoll.method.upper() not in ["FEP", "EQU", "NEQ", "ALL"]:
        raise AttributeError(
            "Only `FEP`, `EQU`, `NEQ` or 'ALL'  are supported methods for endstate corrections"
        )
    if protocoll.method.upper() in [
        "FEP",
        "NEQ",
    ] and protocoll.direction.lower() not in ["bidirectional", "unidirectional"]:
        raise AttributeError(
            "Only `bidirectional` or `unidirectional` protocolls are supported"
        )

    sim = protocoll.sim
    # initialize Results with default values
    r = Results()
    if protocoll.method.upper() == "FEP" or protocoll.method.upper() == "ALL":
        ####################################################
        # ------------------- FEP ---------------------------
        ####################################################
        lambs = np.linspace(0, 1, 2)
        if (
            protocoll.direction.lower() == "bidirectional"
            or protocoll.method.upper() == "ALL"
        ):
            ####################################################
            # ------------------- bidirectional-----------------
            # perform switching from mm to qml
            dEs_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocoll.trajectories[0],
                    nr_of_switches=protocoll.nr_of_switches,
                )[0]
                / kBT
            )
            # perform switching from qml to mm
            dEs_from_qml_to_mm = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocoll.trajectories[-1],
                    nr_of_switches=protocoll.nr_of_switches,
                )[0]
                / kBT
            )

            # set results
            r.dE_mm_to_qml = dEs_from_mm_to_qml
            r.dE_qml_to_mm = dEs_from_qml_to_mm
        elif protocoll.direction == "unidirectional":
            ####################################################
            # ------------------- unidirectional----------------
            # perform switching from mm to qml
            dEs_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocoll.trajectories[0],
                    nr_of_switches=protocoll.nr_of_switches,
                )[0]
                / kBT
            )
            r.dE_mm_to_qml = dEs_from_mm_to_qml
        else:
            raise RuntimeError()

    elif protocoll.method == "NEQ":
        ####################################################
        # ------------------- NEQ ---------------------------
        ####################################################
        lambs = np.linspace(0, 1, protocoll.neq_switching_length)
        if protocoll.direction == "bidirectional":
            ####################################################
            # ------------------- bidirectional-----------------
            # perform switching from mm to qml
            Ws_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocoll.trajectories[0],
                    nr_of_switches=protocoll.nr_of_switches,
                )[0]
                / kBT
            )
            # perform switching from qml to mm
            Ws_from_qml_to_mm = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocoll.trajectories[-1],
                    nr_of_switches=protocoll.nr_of_switches,
                )[0]
                / kBT
            )

            r.W_mm_to_qml = Ws_from_mm_to_qml
            r.W_qml_to_mm = Ws_from_qml_to_mm

        elif protocoll.direction == "unidirectional":
            ####################################################
            # ------------------- unidirectional----------------
            # perform switching from mm to qml
            Ws_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocoll.trajectories[0],
                    nr_of_switches=protocoll.nr_of_switches,
                )[0]
                / kBT
            )
            r.W_mm_to_qml = Ws_from_mm_to_qml

        else:
            raise RuntimeError()
    elif protocoll.method == "EQU":
        ####################################################
        # ------------------- EQU ---------------------------
        ####################################################
        from pymbar import MBAR
        from endstate_correction.equ import calculate_u_kn

        lambs = np.linspace(0, 1, protocoll.equ_every_nth_frame)
        N_k, u_kn = calculate_u_kn(
            protocoll.trajectories,
            sim=protocoll.sim,
            every_nth_frame=protocoll.equ_every_nth_frame,
        )
        r.equ_mbar = MBAR(u_kn, N_k)

    return r
