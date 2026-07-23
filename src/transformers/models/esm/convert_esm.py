
import argparse
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory

import esm as esm_module
import torch
from esm.esmfold.v1.misc import batch_encode_sequences as esmfold_encode_sequences
from esm.esmfold.v1.pretrained import esmfold_v1

from transformers.models.esm.configuration_esm import EsmConfig, EsmFoldConfig
from transformers.models.esm.modeling_esm import (
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmIntermediate,
    EsmLayer,
    EsmOutput,
    EsmSelfAttention,
    EsmSelfOutput,
)
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
from transformers.models.esm.tokenization_esm import EsmTokenizer
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_DATA = [
    (
        "protein1",
        "MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVLGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLAGWSRYIPEGLQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIIIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWVPYASVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASATVSKTETSQVAPA",
    ),
    ("protein2", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"),
    ("protein3", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG"),
    ("protein4", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLA"),
]

MODEL_MAPPING = {
    "esm1b_t33_650M_UR50S": esm_module.pretrained.esm1b_t33_650M_UR50S,
    "esm1v_t33_650M_UR90S_1": esm_module.pretrained.esm1v_t33_650M_UR90S_1,
    "esm1v_t33_650M_UR90S_2": esm_module.pretrained.esm1v_t33_650M_UR90S_2,
    "esm1v_t33_650M_UR90S_3": esm_module.pretrained.esm1v_t33_650M_UR90S_3,
    "esm1v_t33_650M_UR90S_4": esm_module.pretrained.esm1v_t33_650M_UR90S_4,
    "esm1v_t33_650M_UR90S_5": esm_module.pretrained.esm1v_t33_650M_UR90S_5,
    "esm2_t48_15B_UR50D": esm_module.pretrained.esm2_t48_15B_UR50D,
    "esm2_t36_3B_UR50D": esm_module.pretrained.esm2_t36_3B_UR50D,
    "esm2_t33_650M_UR50D": esm_module.pretrained.esm2_t33_650M_UR50D,
    "esm2_t30_150M_UR50D": esm_module.pretrained.esm2_t30_150M_UR50D,
    "esm2_t12_35M_UR50D": esm_module.pretrained.esm2_t12_35M_UR50D,
    "esm2_t6_8M_UR50D": esm_module.pretrained.esm2_t6_8M_UR50D,
    "esmfold_v1": esmfold_v1,
}

restypes = list("ARNDCQEGHILKMFPSTWYV")

restypes_with_x = restypes + ["X"]
restypes_with_extras = restypes_with_x + ["<pad>", "<mask>", "<cls>", "<sep>", "<eos>"]


def get_esmfold_tokenizer():
    pass


def transfer_and_check_weights(original_module, our_module):
    pass


def convert_esm_checkpoint_to_pytorch(
    model: str, pytorch_dump_folder_path: str, classification_head: bool, push_to_repo: str, auth_token: str
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    parser.add_argument("--model", default=None, type=str, required=True, help="Name of model to convert.")
    parser.add_argument("--push_to_repo", type=str, help="Repo to upload to (including username!).")
    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token.")
    args = parser.parse_args()
    convert_esm_checkpoint_to_pytorch(
        args.model, args.pytorch_dump_folder_path, args.classification_head, args.push_to_repo, args.auth_token
    )
