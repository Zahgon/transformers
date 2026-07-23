
import argparse
from io import BytesIO
from pathlib import Path

import av
import httpx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    GitConfig,
    GitForCausalLM,
    GitProcessor,
    GitVisionConfig,
    VideoMAEImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_git_config(model_name):
    pass


def create_rename_keys(config, prefix=""):
    rename_keys = []

    rename_keys.append(
        (f"{prefix}image_encoder.class_embedding", "git.image_encoder.vision_model.embeddings.class_embedding")
    )
    rename_keys.append(
        (
            f"{prefix}image_encoder.positional_embedding",
            "git.image_encoder.vision_model.embeddings.position_embedding.weight",
        )
    )
    rename_keys.append(
        (f"{prefix}image_encoder.conv1.weight", "git.image_encoder.vision_model.embeddings.patch_embedding.weight")
    )
    rename_keys.append((f"{prefix}image_encoder.ln_pre.weight", "git.image_encoder.vision_model.pre_layrnorm.weight"))
    rename_keys.append((f"{prefix}image_encoder.ln_pre.bias", "git.image_encoder.vision_model.pre_layrnorm.bias"))
    rename_keys.append(
        (f"{prefix}image_encoder.ln_post.weight", "git.image_encoder.vision_model.post_layernorm.weight")
    )
    rename_keys.append((f"{prefix}image_encoder.ln_post.bias", "git.image_encoder.vision_model.post_layernorm.bias"))
    rename_keys.append((f"{prefix}image_encoder.proj", "git.image_encoder.visual_projection.weight"))

    for i in range(config.vision_config.num_hidden_layers):
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.bias"))

    rename_keys.append((f"{prefix}textual.embedding.words.weight", "git.embeddings.word_embeddings.weight"))
    rename_keys.append((f"{prefix}textual.embedding.positions.weight", "git.embeddings.position_embeddings.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.0.weight", "git.visual_projection.visual_projection.0.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.0.bias", "git.visual_projection.visual_projection.0.bias"))
    rename_keys.append((f"{prefix}textual.visual_projection.1.weight", "git.visual_projection.visual_projection.1.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.1.bias", "git.visual_projection.visual_projection.1.bias"))

    rename_keys.append((f"{prefix}textual.embedding.layer_norm.weight", "git.embeddings.LayerNorm.weight"))
    rename_keys.append((f"{prefix}textual.embedding.layer_norm.bias", "git.embeddings.LayerNorm.bias"))
    rename_keys.append((f"{prefix}textual.output.weight", "output.weight"))
    rename_keys.append((f"{prefix}textual.output.bias", "output.bias"))
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.weight", f"git.encoder.layer.{i}.attention.self.query.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.bias", f"git.encoder.layer.{i}.attention.self.query.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.weight", f"git.encoder.layer.{i}.attention.self.key.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.bias", f"git.encoder.layer.{i}.attention.self.key.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.weight", f"git.encoder.layer.{i}.attention.self.value.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.bias", f"git.encoder.layer.{i}.attention.self.value.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.weight", f"git.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.bias", f"git.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.weight", f"git.encoder.layer.{i}.attention.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.bias", f"git.encoder.layer.{i}.attention.output.LayerNorm.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.weight", f"git.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.bias", f"git.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.weight", f"git.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.bias", f"git.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.weight", f"git.encoder.layer.{i}.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.bias", f"git.encoder.layer.{i}.output.LayerNorm.bias"))

    if config.num_image_with_embedding is not None:
        rename_keys.append(("img_temperal_embedding.0", "git.img_temperal_embedding.0"))
        rename_keys.append(("img_temperal_embedding.1", "git.img_temperal_embedding.1"))
        rename_keys.append(("img_temperal_embedding.2", "git.img_temperal_embedding.2"))
        rename_keys.append(("img_temperal_embedding.3", "git.img_temperal_embedding.3"))
        rename_keys.append(("img_temperal_embedding.4", "git.img_temperal_embedding.4"))
        rename_keys.append(("img_temperal_embedding.5", "git.img_temperal_embedding.5"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val.T if "image_encoder.visual_projection" in new else val


def read_in_q_k_v(state_dict, config, prefix=""):
    dim = config.vision_config.hidden_size
    for i in range(config.vision_config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_bias")
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[
            :dim, :
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:dim]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            dim : dim * 2, :
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[
            dim : dim * 2
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[
            -dim:, :
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-dim:]


def prepare_img(model_name):
    if "textvqa" in model_name:
        filepath = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
        image = Image.open(filepath).convert("RGB")
    else:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        with httpx.stream("GET", url) as response:
            image = Image.open(BytesIO(response.read()))

    return image


def prepare_video():
    def read_video_pyav(container, indices):
        """
        Decode the video with PyAV decoder.

        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`list[int]`): List of frame indices to decode.

        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        """
        Sample a given number of frame indices from the video.

        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.

        Returns:
            indices (`list[int]`): List of sampled frame indices
        """
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    np.random.seed(0)

    file_path = hf_hub_download(repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset")
    with av.open(file_path) as container:
        num_frames = 6
        indices = sample_frame_indices(
            clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
        )
        frames = read_video_pyav(container, indices)

        return frames


@torch.no_grad()
def convert_git_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="git-base",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )

    args = parser.parse_args()
    convert_git_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
