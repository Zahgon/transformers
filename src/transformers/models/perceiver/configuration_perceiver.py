
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="deepmind/language-perceiver")
@strict
class PerceiverConfig(PreTrainedConfig):

    model_type = "perceiver"

    num_latents: int = 256
    d_latents: int = 1280
    d_model: int = 768
    num_blocks: int = 1
    num_self_attends_per_block: int = 26
    num_self_attention_heads: int = 8
    num_cross_attention_heads: int = 8
    qk_channels: int | None = None
    v_channels: int | None = None
    cross_attention_shape_for_attention: str = "kv"
    self_attention_widening_factor: int = 1
    cross_attention_widening_factor: int = 1
    hidden_act: str = "gelu"
    attention_probs_dropout_prob: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_query_residual: bool = True
    vocab_size: int = 262
    max_position_embeddings: int = 2048
    image_size: int | list[int] | tuple[int, int] = 56
    train_size: list[int] | tuple[int, ...] = (368, 496)
    num_frames: int = 16
    audio_samples_per_frame: int = 1920
    samples_per_patch: int = 16
    output_shape: list[int] | tuple[int, ...] = (1, 16, 224, 224)
    output_num_channels: int = 512
    _label_trainable_num_channels: int = 1024


__all__ = ["PerceiverConfig"]
