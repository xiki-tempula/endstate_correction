from pathlib import Path

import pandas as pd
import pytest
from importlib_resources import files

import endstate_correction
from endstate_correction.protocol import BSSProtocol
from endstate_correction.simulation import EndstateCorrectionCHARMM, EndstateCorrectionAMBER
from endstate_correction.simulation.base import EndstateCorrectionBase
from endstate_correction.topology import CHARMMTopology, AMBERTopology


class TestEndstateCorrectionCharmm():
    @staticmethod
    @pytest.fixture(scope="module")
    def bss_protocol():
        protocol = BSSProtocol(
            timestep=1,
            runtime=0.0001,  # 10 * 10 steps
            temperature=300,
            pressure=1,
            report_interval=10,
            restart_interval=50,
            rlist=1,
            collision_rate=1,
            switchDistance=0,
            restart=False,
            lam=pd.Series(data={"ml-lambda": 0}),
        )
        return protocol

    @staticmethod
    @pytest.fixture(scope="module")
    def setup(tmp_path_factory, bss_protocol):
        package_path = files(endstate_correction)
        system_name = "1_octanol"
        env = "waterbox"
        # define the output directory
        output_base = tmp_path_factory.mktemp(system_name)
        parameter_base = package_path / "data" / "jctc_data"
        # load the charmm specific files (psf, pdb, rtf, prm and str files)
        top = CHARMMTopology(
            psf=str(parameter_base / system_name / "charmm-gui/openmm/step3_input.psf"),
            parameter_set=[
                str(parameter_base / system_name / "charmm-gui/unk/unk.rtf"),
                str(parameter_base / system_name / "charmm-gui/unk/unk.prm"),
                str(parameter_base / "toppar/top_all36_cgenff.rtf"),
                str(parameter_base / "toppar/par_all36_cgenff.prm"),
                str(parameter_base / "toppar/toppar_water_ions.str"),
            ],
            input_config=str(
                parameter_base / system_name / "charmm-gui/input.config.dat"),
            crd=str(parameter_base / system_name / "charmm-gui/openmm/step3_input.pdb"),
        )

        simulation = EndstateCorrectionCHARMM(
            top,
            env=env,
            ml_atoms=list(range(27)),
            protocol=bss_protocol,
            name=system_name,
            work_dir=output_base,
        )
        simulation.start()
        return simulation

    def test_sanity(self, setup):
        """Test if the setup is fine."""
        assert isinstance(setup, EndstateCorrectionBase)

    def test_dcd(self, setup):
        assert Path(setup._traj_file).is_file()

class TestEndstateCorrectionAmber(TestEndstateCorrectionCharmm):
    @staticmethod
    @pytest.fixture(scope="module")
    def setup(tmp_path_factory, bss_protocol):
        package_path = files(endstate_correction)
        system_name = "methane"
        env = "waterbox"
        # define the output directory
        output_base = tmp_path_factory.mktemp(system_name)
        parameter_base = package_path / "data" / "amber"
        # load the charmm specific files (psf, pdb, rtf, prm and str files)
        top = AMBERTopology(
            prm7=str(parameter_base / f'{system_name}.prm7'),
            rst7=str(parameter_base / f'{system_name}.rst7')
        )

        simulation = EndstateCorrectionAMBER(
            top,
            env=env,
            ml_atoms=list(range(5)),
            protocol=bss_protocol,
            name=system_name,
            work_dir=output_base,
        )
        simulation.start()
        return simulation