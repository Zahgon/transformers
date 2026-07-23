
import argparse
import re

from laion_clap import CLAP_Module

from transformers import AutoFeatureExtractor, ClapConfig, ClapModel


KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

processor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused", truncation="rand_trunc")


def init_clap(checkpoint_path, model_type, enable_fusion=False):
    pass


def get_config_from_original(clap_model):
    pass


def rename_state_dict(state_dict):
    model_state_dict = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer) // 3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("qkv", "query")] = query_layer
            model_state_dict[key.replace("qkv", "key")] = key_layer
            model_state_dict[key.replace("qkv", "value")] = value_layer
        else:
            model_state_dict[key] = value

    return model_state_dict


def convert_clap_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path, model_type, enable_fusion=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--enable_fusion", action="store_true", help="Whether to enable fusion or not")
    parser.add_argument("--model_type", default="HTSAT-tiny", type=str, help="Whether to enable fusion or not")
    args = parser.parse_args()

    convert_clap_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.model_type, args.enable_fusion
    )
