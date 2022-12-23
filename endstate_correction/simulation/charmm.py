from pathlib import Path

from openmm import unit
from openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile, \
    PDBFile

from .base import EndstateCorrectionBase
from ..system import read_box


class EndstateCorrectionCHARMM(EndstateCorrectionBase):
    def _get_mm_topology(self) -> CharmmPsfFile:
        psf = CharmmPsfFile(self.top.psf_file_path)
        # set up the treatment of the system for the specific environment
        if self.env == "waterbox":
            psf = read_box(psf, self.top.input_config)
        return psf

    def _get_initial_coordinates(self) -> unit.Quantity:
        ext = Path(self.top.crd_file_path).suffix.lower()
        if ext == ".pdb":
            coord = CharmmCrdFile(self.top.crd_file_path)
        elif ext == ".crd":
            coord = PDBFile(self.top.crd_file_path)
        else:
            raise NotImplementedError(
                "The file extension {ext} cannot be recognised. Only support (.pdb or .crd)."
            )

        return coord.positions

    def _createSystem(self, **kwargs):
        kwargs = {
            "params": CharmmParameterSet(*self.top.parameter_set),
            "switchDistance": self.protocol.switchDistance * unit.nanometers,
            **kwargs,
        }
        return super()._createSystem(**kwargs)
