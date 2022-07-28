from endstate_correction.utils import convert_pickle_to_dcd_file
import endstate_correction
import pathlib
import glob


def test_converting():
    """Convert pickle trajectory to dcd file"""
    path = pathlib.Path(endstate_correction.__file__).resolve().parent

    system_name = "ZINC00079729"
    path_to_topology = f"{path}/data/hipen_data/{system_name}/{system_name}.psf"
    path_to_coords = f"{path}/data/hipen_data/{system_name}/{system_name}.crd"

    pickle_files = glob.glob(
        f"data/{system_name}/sampling_charmmff/run01/{system_name}*.pickle"
    )
    for p in pickle_files:
        print(p)
        convert_pickle_to_dcd_file(
            pickle_file_path=p,
            path_to_topology=path_to_topology,
            dcd_output_path=f'{".".join(p.split(".")[:-1])}.dcd',
            pdb_output_path=f'{p.split("_")[0].replace("sampling", system_name)}.pdb',
            path_to_coords=path_to_coords,
        )
