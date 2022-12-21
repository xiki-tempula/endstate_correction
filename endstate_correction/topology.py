from dataclasses import dataclass
from typing import List


@dataclass
class AMBERTopology:
    """This is a dataclass for amber topology."""

    rst7: str
    prm7: str


@dataclass
class CHARMMTopology:
    """This is a dataclass for CHARMM topology."""

    psf: str
    crd: str
    parameter_set: List[str]
    input_config: str
