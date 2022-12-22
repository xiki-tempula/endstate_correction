from openmm import unit
from openmm.app import CharmmParameterSet, CharmmPsfFile, PDBFile, CharmmCrdFile

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
        try:
            coord = CharmmCrdFile(self.top.crd_file_path)
        except:
            coord = PDBFile(self.top.crd_file_path)

        return coord.positions

    def _createSystem(self, **kwargs):
        kwargs = {
            "params": CharmmParameterSet(*self.top.parameter_set),
            "switchDistance": self.protocol.switchDistance * unit.nanometers,
            **kwargs,
        }
        return super()._createSystem(**kwargs)
