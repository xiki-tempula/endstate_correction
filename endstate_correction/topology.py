from dataclasses import dataclass
from typing import List


@dataclass
class AMBERTopology:
    """This is a dataclass for amber topology."""

    RST7: str
    PRM7: str


@dataclass
class CHARMMTopology:
    """This is a dataclass for CHARMM topology."""

    Psf: str
    Crd: str
    ParameterSet: List[str]
    input_config: str
