"""Provide functions for the endstate correction workflow."""


from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from openmm.app import Simulation
from pymbar import MBAR


@dataclass
class BSSProtocol:
    """This is a dataclass mimicking the BioSimSpace.Protocol."""

    timestep: int  # fs
    runtime: float  # ns
    temperature: float  # K
    pressure: float  # atm
    report_interval: int  # The frequency at which energy are recorded (In integration steps).
    restart_interval: int  # The frequency at which frames are recorded (In integration steps).
    rlist: float  # short-range cutoff nanometers.
    collision_rate: float  # 1/picosecond
    switchDistance: float  # nanometers
    lam: pd.Series  # Current lambda
    restart: bool  # Whether to reset the velocity or not


@dataclass
class Protocol:
    """Defining the endstate correction protocol"""

    method: str
    sim: Simulation
    trajectories: List
    direction: str = "None"
    nr_of_switches: int = -1
    neq_switching_length: int = 5_000


@dataclass
class Results:
    """Provides a dataclass containing the results of a protocol"""

    dE_mm_to_qml: np.array = np.array([])
    dE_qml_to_mm: np.array = np.array([])
    W_mm_to_qml: np.array = np.array([])
    W_qml_to_mm: np.array = np.array([])
    equ_mbar: MBAR = None


def perform_endstate_correction(protocol: Protocol) -> Results:
    """Perform endstate correction using the provided protocol.

    Args:
        protocol (Protocol): defines the endstatte correction

    Raises:
        AttributeError: _description_
        AttributeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        Results: results generated using the passed protocol
    """

    from endstate_correction.neq import perform_switching
    from endstate_correction.constant import kBT

    print(protocol.method)
    # check that all necessary keywords are present
    if protocol.method.upper() not in ["FEP", "NEQ", "ALL"]:
        raise AttributeError(
            "Only `FEP`, 'NEQ` or 'ALL'  are supported methods for endstate corrections"
        )
    if protocol.method.upper() in [
        "FEP",
        "NEQ",
    ] and protocol.direction.lower() not in ["bidirectional", "unidirectional"]:
        raise AttributeError(
            "Only `bidirectional` or `unidirectional` protocols are supported"
        )

    sim = protocol.sim
    # initialize Results with default values
    r = Results()
    if protocol.method.upper() == "FEP" or protocol.method.upper() == "ALL":
        ####################################################
        # ------------------- FEP ---------------------------
        ####################################################
        print("#####################################################")
        print("# ------------------- FEP ---------------------------")
        print("#####################################################")
        # from MM to QML
        if (
            protocol.direction.lower() == "bidirectional"
            or protocol.method.upper() == "ALL"
        ):
            ####################################################
            # ------------------- bidirectional-----------------
            # perform switching from mm to qml

            assert len(protocol.trajectories) == 2

            print("Performing bidirectional protocol ...")
            lambs = np.linspace(0, 1, 2)
            dEs_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                )[0]
                / kBT
            )
            # perform switching from qml to mm
            lambs = np.linspace(1, 0, 2)
            dEs_from_qml_to_mm = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[-1],
                    nr_of_switches=protocol.nr_of_switches,
                )[0]
                / kBT
            )

            # set results
            r.dE_mm_to_qml = dEs_from_mm_to_qml
            r.dE_qml_to_mm = dEs_from_qml_to_mm
        elif protocol.direction == "unidirectional":
            ####################################################
            # ------------------- unidirectional----------------
            # perform switching from mm to qml
            print("Performing unidirectional protocol ...")
            lambs = np.linspace(0, 1, 2)
            dEs_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                )[0]
                / kBT
            )
            r.dE_mm_to_qml = dEs_from_mm_to_qml
        else:
            raise RuntimeError()

    if protocol.method.upper() == "NEQ" or protocol.method.upper() == "ALL":
        ####################################################
        # ------------------- NEQ ---------------------------
        ####################################################
        print("#####################################################")
        print("# ------------------- NEQ ---------------------------")
        print("#####################################################")
        if protocol.direction == "bidirectional" or protocol.method.upper() == "ALL":
            ####################################################
            # ------------------- bidirectional-----------------
            # perform switching from mm to qml

            assert len(protocol.trajectories) == 2

            print("Performing bidirectional protocol ...")
            lambs = np.linspace(0, 1, protocol.neq_switching_length)
            Ws_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                )[0]
                / kBT
            )
            # perform switching from qml to mm
            lambs = np.linspace(1, 0, protocol.neq_switching_length)
            Ws_from_qml_to_mm = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[-1],
                    nr_of_switches=protocol.nr_of_switches,
                )[0]
                / kBT
            )

            r.W_mm_to_qml = Ws_from_mm_to_qml
            r.W_qml_to_mm = Ws_from_qml_to_mm

        elif protocol.direction == "unidirectional":
            ####################################################
            # ------------------- unidirectional----------------
            # perform switching from mm to qml
            print("Performing unidirectional protocol ...")
            lambs = np.linspace(0, 1, protocol.neq_switching_length)
            Ws_from_mm_to_qml = np.array(
                perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                )[0]
                / kBT
            )
            r.W_mm_to_qml = Ws_from_mm_to_qml

        else:
            raise RuntimeError()

    return r
