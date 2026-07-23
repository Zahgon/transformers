
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V3.2-Exp")
@strict
class DeepseekV32Config(Glm4MoeLiteConfig, RotaryEmbeddingConfigMixin):

    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
    mlp_bias: bool = False
    head_dim: int = 64
    first_k_dense_replace: int = 3
    pretraining_tp = AttributeError()
    rope_interleave = AttributeError()
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.num_hidden_layers - n_dense)
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        if (num_experts := kwargs.get("num_experts")) is not None:
            self.n_routed_experts = num_experts

        super().__post_init__(**kwargs)


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class DeepseekV32Indexer(nn.Module):

    def __init__(self, config: "DeepseekV32Config", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        This is the bf16 equivalent of the reference Indexer which uses `rotate_activation` (Hadamard transform)
        and `fp8_index` (FP8 quantized scoring kernel). Since the Hadamard transform is orthogonal (dot products
        are preserved: Hq·Hk = q·k), and FP8 quantization is a precision optimization, we skip both and compute
        scores directly in bf16/fp32.

        The scoring logic computes:
            index_score[b,s,t] = Σ_h (weight[b,s,h] · softmax_scale · q[b,s,h,:] · k[b,t,:])

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            past_key_values: Cache object containing the indexer key cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]`. The eager / SDPA paths
                turn these into an additive sparse mask; the `flash-mla` kernel consumes them directly.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)

        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask
        else:
            key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
            causal = key_positions[None, None, :] > position_ids[:, :, None]  # [B, S, T]
            index_scores = index_scores.masked_fill(causal, float("-inf"))

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class DeepseekV32Attention(DeepseekV3Attention):

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
        topk_indices = self.indexer(
            hidden_states, q_resid, position_embeddings, indexer_mask, position_ids, past_key_values=past_key_values
        )  # [B, S, topk]

        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
            index_mask = (
                topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )  # [B, 1, S, T]; True = masked
            if attention_mask is None:
                key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
                index_mask = index_mask | (key_positions[None, None, None, :] > position_ids[:, None, :, None])
                attention_mask = hidden_states.new_zeros((batch_size, 1, seq_length, key_states.shape[2]))
            attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        else:
            sparse_indices = topk_indices

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            indices=sparse_indices,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False


class DeepseekV32Model(DeepseekV3Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"deepseek_sparse_attention": create_causal_mask(**mask_kwargs)}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
]
