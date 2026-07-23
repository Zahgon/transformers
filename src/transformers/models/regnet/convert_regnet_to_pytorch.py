
import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import timm
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams, RegNetY32gf, RegNetY64gf, RegNetY128gf
from huggingface_hub import hf_hub_download
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs

from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger()


@dataclass
class Tracker:
    module: nn.Module
    traced: list[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        pass

    def __call__(self, x: Tensor):
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        [x.remove() for x in self.handles]
        return self

    @property
    def parametrized(self):
        pass


@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 1
    src_skip: list = field(default_factory=list)
    dest_skip: list = field(default_factory=list)
    raise_if_mismatch: bool = True

    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized

        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        if len(dest_traced) != len(src_traced) and self.raise_if_mismatch:
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f"Transferred from={src_m} to={dest_m}")


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


class NameToFromModelFuncMap(dict):

    def convert_name_to_timm(self, x: str) -> str:
        x_split = x.split("-")
        return x_split[0] + x_split[1] + "_" + "".join(x_split[2:])

    def __getitem__(self, x: str) -> Callable[[], tuple[nn.Module, dict]]:
        if x not in self:
            x = self.convert_name_to_timm(x)
            val = partial(lambda: (timm.create_model(x, pretrained=True).eval(), None))

        else:
            val = super().__getitem__(x)

        return val


class NameToOurModelFuncMap(dict):

    def __getitem__(self, x: str) -> Callable[[], nn.Module]:
        if "seer" in x and "in1k" not in x:
            val = RegNetModel
        else:
            val = RegNetForImageClassification
        return val


def manually_copy_vissl_head(from_state_dict, to_state_dict, keys: list[tuple[str, str]]):
    pass


def convert_weight_and_push(
    name: str,
    from_model_func: Callable[[], nn.Module],
    our_model_func: Callable[[], nn.Module],
    config: RegNetConfig,
    save_directory: Path,
    push_to_hub: bool = True,
):
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
