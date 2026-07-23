

import dataclasses
import re
import string
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from . import residue_constants


FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01


@dataclasses.dataclass(frozen=True)
class Protein:

    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    aatype: np.ndarray  # [num_res]

    atom_mask: np.ndarray  # [num_res, num_atom_type]

    residue_index: np.ndarray  # [num_res]

    b_factors: np.ndarray  # [num_res, num_atom_type]

    chain_index: np.ndarray | None = None

    remark: str | None = None

    parents: Sequence[str] | None = None

    parents_chain_index: Sequence[int] | None = None


def from_proteinnet_string(proteinnet_str: str) -> Protein:
    pass


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> list[str]:
    pass


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    pass


def to_pdb(prot: Protein) -> str:
    pass


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    pass


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    remark: str | None = None,
    parents: Sequence[str] | None = None,
    parents_chain_index: Sequence[int] | None = None,
) -> Protein:
    pass
