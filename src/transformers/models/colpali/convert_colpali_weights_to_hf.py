
import argparse
import glob
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import AutoConfig
from transformers.models.colpali import ColPaliForRetrieval
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_DTYPE = torch.bfloat16


def rename_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    pass


def load_original_state_dict(model_id: str, revision: str | None = None) -> dict[str, torch.Tensor]:
    directory_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["*.safetensors"],
    )

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    if "lm_head.weight" not in original_state_dict:
        original_state_dict["vlm.language_model.lm_head.weight"] = original_state_dict[
            "model.language_model.model.embed_tokens.weight"
        ].clone()

    return original_state_dict


@torch.no_grad()
def convert_colpali_weights_to_hf(
    model_id: str,
    output_dir: str,
    push_to_hub: bool,
    revision: str | None = None,
    original_vlm_name_or_path: str | None = None,
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This script converts the original ColPali model to the HF model format.

        Example usage:
        ```bash
        python src/transformers/models/colpali/convert_colpali_weights_to_hf.py \
            --model_id vidore/colpali-v1.2-merged \
            --revision 89fd9736194236a1ecb7a9ec9b04f537f6f896af \
            --original_vlm_name_or_path google/paligemma-3b-mix-448 \
            --output_dir vidore/colpali-v1.2-hf \
            --push_to_hub
        ```
        """
    )
    parser.add_argument(
        "--model_id",
        help="Model ID of the original model to convert",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--revision",
        help="Revision of the model to download",
        default=None,
    )
    parser.add_argument(
        "--original_vlm_name_or_path",
        help="Name or path of the original VLM backbone model",
        default=None,
    )
    args = parser.parse_args()

    convert_colpali_weights_to_hf(
        model_id=args.model_id,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        revision=args.revision,
        original_vlm_name_or_path=args.original_vlm_name_or_path,
    )
