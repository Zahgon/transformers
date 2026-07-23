
from typing import Union

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ...utils.type_validators import interval, is_divisible_by


logger = logging.get_logger(__name__)


@strict
class StructureModuleConfig(PreTrainedConfig):

    sequence_dim: int | None = 384
    pairwise_dim: int | None = 128
    ipa_dim: int | None = 16
    resnet_dim: int | None = 128
    num_heads_ipa: int | None = 12
    num_qk_points: int | None = 4
    num_v_points: int | None = 8
    dropout_rate: float | None = 0.1
    num_blocks: int | None = 8
    num_transition_layers: int | None = 1
    num_resnet_blocks: int | None = 2
    num_angles: int | None = 7
    trans_scale_factor: int | None = 10
    epsilon: float | None = 1e-8
    inf: float | None = 1e5


@strict
class TrunkConfig(PreTrainedConfig):
    sub_configs = {"structure_module": StructureModuleConfig}

    num_blocks: int | None = 48
    sequence_state_dim: int | None = 1024
    pairwise_state_dim: int | None = is_divisible_by(divisor=2)(default=128)
    sequence_head_width: int | None = 32
    pairwise_head_width: int | None = 32
    position_bins: int | None = 32
    dropout: float | int | None = interval(max=0.4)(default=0.0)
    layer_drop: float | int | None = 0.0
    cpu_grad_checkpoint: bool | None = False
    max_recycles: int | None = interval(min=0)(default=4)
    chunk_size: int | None = 128
    structure_module: Union[dict, "StructureModuleConfig"] | None = None

    def __post_init__(self, **kwargs):
        if self.structure_module is None:
            self.structure_module = StructureModuleConfig()
        elif isinstance(self.structure_module, dict):
            self.structure_module = StructureModuleConfig(**self.structure_module)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@strict
class EsmFoldConfig(PreTrainedConfig):
    sub_configs = {"trunk": TrunkConfig}

    esm_type: str | None = None
    fp16_esm: bool | None = True
    use_esm_attn_map: bool | None = False
    esm_ablate_pairwise: bool | None = False
    esm_ablate_sequence: bool | None = False
    esm_input_dropout: float | int | None = 0.0
    embed_aa: bool | None = True
    bypass_lm: bool | None = False
    lddt_head_hid_dim: int | None = 128
    trunk: Union[dict, "TrunkConfig"] | None = None

    def __post_init__(self, **kwargs):
        if self.trunk is None:
            self.trunk = TrunkConfig()
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/esm-1b")
@strict
class EsmConfig(PreTrainedConfig):

    model_type = "esm"
    sub_configs = {"esmfold_config": EsmFoldConfig}

    vocab_size: int | None = None
    mask_token_id: int | None = None
    pad_token_id: int | None = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float | None = 0.1
    attention_probs_dropout_prob: float | None = 0.1
    max_position_embeddings: int = 1026
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    layer_norm_eps: float | None = 1e-12
    position_embedding_type: str | None = "absolute"
    use_cache: bool = True
    emb_layer_norm_before: bool | None = None
    token_dropout: bool | None = False
    is_folding_model: bool | None = False
    esmfold_config: dict | EsmFoldConfig | None = None
    vocab_list: list[str] | tuple[str, ...] | None = None
    is_decoder: bool | None = False
    add_cross_attention: bool | None = False
    tie_word_embeddings: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        if self.is_folding_model:
            if self.esmfold_config is None:
                logger.info("No esmfold_config supplied for folding model, using default values.")
                self.esmfold_config = EsmFoldConfig()
            elif isinstance(self.esmfold_config, dict):
                self.esmfold_config = EsmFoldConfig(**self.esmfold_config)

            if self.vocab_list is None:
                logger.warning("No vocab_list supplied for folding model, assuming the ESM-2 vocabulary!")
                self.vocab_list = get_default_vocab_list()
        else:
            self.esmfold_config = None
            self.vocab_list = None

        if self.esmfold_config is not None and getattr(self.esmfold_config, "use_esm_attn_map", False):
            raise ValueError("The HuggingFace port of ESMFold does not support use_esm_attn_map at this time!")

        super().__post_init__(**kwargs)


def get_default_vocab_list():
    return (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "<null_1>",
        "<mask>",
    )


__all__ = ["EsmConfig"]
