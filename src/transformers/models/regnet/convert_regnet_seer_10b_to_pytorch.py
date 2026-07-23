
import argparse
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams
from huggingface_hub import hf_hub_download
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs

from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.modeling_utils import _load_state_dict_into_meta_model, load_state_dict
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger()


@dataclass
class Tracker:
    module: nn.Module
    traced: list[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)
    name2module: dict[str, nn.Module] = field(default_factory=OrderedDict)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor, name: str):
        pass

    def __call__(self, x: Tensor):
        for name, m in self.module.named_modules():
            self.handles.append(m.register_forward_hook(partial(self._forward_hook, name=name)))
        self.module(x)
        [x.remove() for x in self.handles]
        return self

    @property
    def parametrized(self):
        pass


class FakeRegNetVisslWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()

        feature_blocks: list[tuple[str, nn.Module]] = []
        feature_blocks.append(("conv1", model.stem))
        for k, v in model.trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x: Tensor):
        return get_trunk_forward_outputs(
            x,
            out_feat_keys=None,
            feature_blocks=self._feature_blocks,
        )


class FakeRegNetParams(RegNetParams):

    def get_expanded_params(self):
        pass


def get_from_to_our_keys(model_name: str) -> dict[str, str]:
    pass


def convert_weights_and_push(save_directory: Path, model_name: str | None = None, push_to_hub: bool = True):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported regnet* architecture,"
            " currently: regnetx-*, regnety-*. If `None`, all of them will the converted."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )

    args = parser.parse_args()

    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
