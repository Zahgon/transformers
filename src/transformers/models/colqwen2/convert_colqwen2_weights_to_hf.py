
import argparse
import glob
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open

from transformers import AutoConfig, AutoModel
from transformers.models.colqwen2 import ColQwen2ForRetrieval
from transformers.models.colqwen2.configuration_colqwen2 import ColQwen2Config
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_DTYPE = torch.bfloat16


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

    if "lm_head.weight" not in original_state_dict and "model.embed_tokens.weight" in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    return original_state_dict


def rename_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    pass


@torch.no_grad()
def convert_colqwen2_weights_to_hf(
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
        This script converts the original ColQwen2 model to the HF model format.

        Don't forget to manually upload the processor-related files to the HF model repository
        after running this script.

        Example usage:
        ```bash
        python src/transformers/models/colqwen2/convert_colqwen2_weights_to_hf.py \
            --model_id vidore/colqwen2-v1.0-merged \
            --revision eeccbae1d44bdcb0c83b1788127a2b2cad7d718e \
            --original_vlm_name_or_path Qwen/Qwen2-VL-2B-Instruct \
            --output_dir vidore/colqwen2-v1.0-hf-internal \
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

    convert_colqwen2_weights_to_hf(
        model_id=args.model_id,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        revision=args.revision,
        original_vlm_name_or_path=args.original_vlm_name_or_path,
    )
