
import argparse

from torch import nn

from transformers_old.modeling_prophetnet import (
    ProphetNetForConditionalGeneration as ProphetNetForConditionalGenerationOld,
)
from transformers_old.modeling_xlm_prophetnet import (
    XLMProphetNetForConditionalGeneration as XLMProphetNetForConditionalGenerationOld,
)

from transformers import ProphetNetForConditionalGeneration, XLMProphetNetForConditionalGeneration, logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def convert_prophetnet_checkpoint_to_pytorch(prophetnet_checkpoint_path: str, pytorch_dump_folder_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prophetnet_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_prophetnet_checkpoint_to_pytorch(args.prophetnet_checkpoint_path, args.pytorch_dump_folder_path)
