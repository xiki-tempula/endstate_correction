import pytest
from importlib_resources import files

import endstate_correction
from endstate_correction.protocol import BSSProtocol
from endstate_correction.simulation import EndstateCorrectionCHARMM
from endstate_correction.topology import CHARMMTopology


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    package_path = files(endstate_correction)
    system_name = "1_octanol"
    output_base = tmp_path_factory.mktemp(system_name)
    env = "waterbox"
    # define the output directory
    output_base = system_name
    parameter_base = package_path / "data" / "jctc_data"
    # load the charmm specific files (psf, pdb, rtf, prm and str files)
    top = CHARMMTopology(
        Psf=str(parameter_base / system_name / "charmm-gui/openmm/step3_input.psf"),
        ParameterSet=[
            str(parameter_base / system_name / "charmm-gui/unk/unk.rtf"),
            str(parameter_base / system_name / "charmm-gui/unk/unk.prm"),
            str(parameter_base / "toppar/top_all36_cgenff.rtf"),
            str(parameter_base / "toppar/par_all36_cgenff.prm"),
            str(parameter_base / "toppar/toppar_water_ions.str"),
        ],
        input_config=str(parameter_base / system_name / "charmm-gui/input.config.dat"),
        Crd=str(parameter_base / system_name / "charmm-gui/openmm/step3_input.pdb"),
    )
    protocol = BSSProtocol(
        timestep=2,
        runtime=1,
        temperature=300,
        pressure=1,
        report_interval=1000,
        restart_interval=1000,
        rlist=1,
        collision_rate=1,
        switchDistance=0,
    )
    simulation = EndstateCorrectionCHARMM(
        top, env=env, ml_atoms=list(range(27)), protocol=protocol
    )
    return simulation


def test_sanity(setup):
    pass
