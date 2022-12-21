from openmm.app import AmberPrmtopFile, AmberInpcrdFile

from .base import EndstateCorrectionBase


class EndstateCorrectionAMBER(EndstateCorrectionBase):
    def _get_mm_topology(self) -> AmberPrmtopFile:
        psf = AmberPrmtopFile(self.top.PRM7)
        return psf

    def _get_mm_coordinate(self) -> AmberInpcrdFile:
        return AmberInpcrdFile(self.top.RST7)


