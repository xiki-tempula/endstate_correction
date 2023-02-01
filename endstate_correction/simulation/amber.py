from openmm.app import AmberInpcrdFile, AmberPrmtopFile

from .base import EndstateCorrectionBase


class EndstateCorrectionAMBER(EndstateCorrectionBase):
    def _get_mm_topology(self) -> AmberPrmtopFile:
        prm7 = AmberPrmtopFile(self.top.prm7_file_path)
        return prm7

    def _get_initial_coordinates(self) -> AmberInpcrdFile:
        return AmberInpcrdFile(self.top.rst7_file_path).positions
