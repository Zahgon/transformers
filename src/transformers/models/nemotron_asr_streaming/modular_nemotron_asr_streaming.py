
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...feature_extraction_utils import BatchFeature
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..fastspeech2_conformer.modeling_fastspeech2_conformer import FastSpeech2ConformerConvolutionModule
from ..llama.modeling_llama import eager_attention_forward
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig, ParakeetRNNTConfig
from ..parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from ..parakeet.modeling_parakeet import (
    ParakeetEncoder,
    ParakeetEncoderAttention,
    ParakeetEncoderBlock,
    ParakeetEncoderRelPositionalEncoding,
    ParakeetForRNNT,
    ParakeetPreTrainedModel,
    ParakeetRNNTDecoder,
    ParakeetRNNTJointNetwork,
    ParakeetRNNTOutput,
)
from ..voxtral_realtime.modeling_voxtral_realtime import (
    VoxtralRealtimeCausalConv1d,
    VoxtralRealtimeConv1dCacheLayer,
)
from .generation_nemotron_asr_streaming import (
    NemotronAsrStreamingGenerationMixin,
    NemotronAsrStreamingRNNTDecoderCache,
)


LOG_ZERO_GUARD_VALUE = 2**-24

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrStreamingEncoderConfig(ParakeetEncoderConfig):

    sliding_window: int = 71
    default_num_lookahead_tokens: int = 13

    @property
    def subsampling_out_hidden_size(self) -> int:
        pass


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrStreamingConfig(ParakeetRNNTConfig):

    model_type = "nemotron_asr_streaming"
    vocab_size: int = 1025
    pad_token_id: int = 0
    blank_token_id: int = 1024


class NemotronAsrStreamingFeatureExtractor(ParakeetFeatureExtractor):
    def _torch_extract_fbank_features(self, waveform, device="cpu", center=True):
        pass

    def __call__(
        self,
        raw_speech: "np.ndarray | list[float] | list[np.ndarray] | list[list[float]]",
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: "str | TensorType | None" = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        device: str | None = "cpu",
        return_token_timestamps: bool | None = None,
        center: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). Implementation uses PyTorch for
        the STFT computation if available, otherwise a slower NumPy based one.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram of audio signals in the
                `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")
            return_token_timestamps (`bool`, *optional*, defaults to `None`):
                Deprecated. Use `return_attention_mask` instead from which the number of frames can be inferred.
            center (`bool`, *optional*, defaults to `True`):
                Whether to pad the audio on both sides so STFT frames are centered (`torch.stft(center=True)`). Use
                `True` for offline extraction and for the first chunk of a streaming session. Use `False` for
                subsequent streaming chunks: feeding `audio[hop * frame - n_fft // 2 : ...]` with `center=False`
                reproduces, frame-for-frame, the features that a single `center=True` pass over the whole utterance
                would have produced for those frames.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple))
        if is_batched_sequence:
            for speech in raw_speech:
                if len(speech.shape) > 1:
                    logger.warning(
                        f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                        "We will take the mean of the channels to convert to mono."
                    )
                    speech = speech.mean(-1)

        if is_batched_torch or is_batched_sequence:
            raw_speech = [speech[:, None].to(torch.float32) for speech in raw_speech]
        else:
            raw_speech = [raw_speech[:, None].to(torch.float32)]

        audio_lengths = [len(speech) for speech in raw_speech]
        batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_features = padded_inputs.input_features.squeeze(-1)

        if self.preemphasis is not None:
            timemask = torch.arange(input_features.shape[1], device=input_features.device).unsqueeze(
                0
            ) < padded_inputs.audio_lengths.unsqueeze(1)
            input_features = torch.cat(
                [input_features[:, :1], input_features[:, 1:] - self.preemphasis * input_features[:, :-1]], dim=1
            )
            input_features = input_features.masked_fill(~timemask, 0.0)

        input_features = self._torch_extract_fbank_features(input_features, device, center=center)
        if center:
            features_lengths = torch.floor_divide(
                padded_inputs.audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length
            )
        else:
            features_lengths = torch.floor_divide(padded_inputs.audio_lengths - self.n_fft, self.hop_length) + 1
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < features_lengths[:, None]

        input_features *= attention_mask.unsqueeze(-1)

        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )


class NemotronAsrStreamingEncoderCausalConv1dCacheLayer(VoxtralRealtimeConv1dCacheLayer): ...


class NemotronAsrStreamingEncoderCausalConv2dCacheLayer:
    def __init__(self):
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(self, hidden_states, conv_module):
        self.left_pad = conv_module.left_pad
        self.init_pad = conv_module.left_pad_init - conv_module.left_pad
        cache_shape = list(hidden_states.shape)
        cache_shape[2] = self.left_pad
        self.cache = torch.zeros(cache_shape, device=hidden_states.device, dtype=hidden_states.dtype)

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.cache)

        self.is_first_chunk = True
        self.is_initialized = True

    def update(self, hidden_states, conv_module=None):
        if not self.is_initialized and conv_module is not None:
            self.lazy_initialization(hidden_states, conv_module)
        elif not self.is_initialized:
            raise ValueError(
                "NemotronAsrStreamingEncoderCausalConv2dCacheLayer is not initialized. Make sure to provide conv_module to the update method."
            )

        shortfall = max(0, self.left_pad - hidden_states.shape[2])
        if shortfall > 0:
            new_cache = torch.cat([self.cache[:, :, -shortfall:], hidden_states], dim=2)
        else:
            new_cache = hidden_states[:, :, -self.left_pad :]

        current_cache = self.cache.clone()
        if self.is_first_chunk and self.init_pad > 0:
            init_shape = list(current_cache.shape)
            init_shape[2] = self.init_pad
            current_cache = torch.cat([current_cache.new_zeros(init_shape), current_cache], dim=2)
        self.is_first_chunk = False

        self.cache.copy_(new_cache)
        return current_cache


class NemotronAsrStreamingEncoderCausalConvPaddingCache:
    def __init__(self):
        self.layers: dict[str, NemotronAsrStreamingEncoderCausalConv1dCacheLayer] = {}

    def update(self, hidden_states, cache_key, conv_module):
        if cache_key not in self.layers:
            if isinstance(conv_module, NemotronAsrStreamingEncoderCausalConv2D):
                self.layers[cache_key] = NemotronAsrStreamingEncoderCausalConv2dCacheLayer()
            elif isinstance(conv_module, NemotronAsrStreamingEncoderCausalConv1d):
                self.layers[cache_key] = NemotronAsrStreamingEncoderCausalConv1dCacheLayer()
            else:
                raise NotImplementedError(f"Unsupported conv_module type: {type(conv_module)}")

        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        return torch.cat([padding_states, hidden_states], dim=2)


class NemotronAsrStreamingEncoderCausalConv1d(VoxtralRealtimeCausalConv1d): ...


class NemotronAsrStreamingEncoderCausalConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        self.cache_key = cache_key

    @property
    def left_pad(self):
        pass

    @property
    def left_pad_init(self):
        pass

    @property
    def time_pad(self):
        pass

    @property
    def freq_pad(self):
        pass

    def output_length(self, input_lengths: torch.Tensor | None, streaming: bool = False) -> torch.Tensor | None:
        if input_lengths is None:
            return None
        left, right = (self.left_pad, 0) if streaming else self.time_pad
        return (input_lengths + left + right - self.kernel_size[0]) // self.stride[0] + 1

    def forward(
        self,
        x: torch.Tensor,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
    ) -> torch.Tensor:
        x = nn.functional.pad(x, (self.freq_pad[0], self.freq_pad[1]))
        if padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = nn.functional.pad(x, (0, 0, self.time_pad[0], self.time_pad[1]))

        return super().forward(x)


@auto_docstring(
    custom_intro="""
    Extends [`ParakeetEncoderModelOutput`] with optional streaming caches. Caches are only populated for
    cache-aware models when `use_cache=True`.
    """
)
@dataclass
class NemotronAsrStreamingEncoderModelOutput(BaseModelOutputWithPooling):

    attention_mask: torch.Tensor | None = None
    past_key_values: Cache | None = None
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None


class NemotronAsrStreamingEncoderRelPositionalEncoding(ParakeetEncoderRelPositionalEncoding):
    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, cached_frames: int | None = None):
        # style relative encoding spans the full key length `L = current chunk + cached_frames`, with
        seq_length = hidden_states.shape[1] + (cached_frames if cached_frames is not None else 0)
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {seq_length} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )
        position_ids = torch.arange(seq_length - 1, -seq_length, -1, device=hidden_states.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(hidden_states.shape[0], -1, 1).to(hidden_states.device)
        )
        position_ids_expanded = position_ids[None, None, :].float()

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            sin = freqs.sin()
            cos = freqs.cos()
            pos_embed = torch.stack([sin, cos], dim=-1)
            pos_embed = pos_embed.reshape(*pos_embed.shape[:-2], -1)

        return pos_embed.to(dtype=hidden_states.dtype)


class NemotronAsrStreamingEncoderConvolutionModule(FastSpeech2ConformerConvolutionModule):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig, module_config=None, layer_idx: int | None = None):
        super().__init__(config, module_config)
        kernel_size = config.conv_kernel_size
        channels = config.hidden_size

        self.norm = nn.LayerNorm(channels)
        self.depthwise_conv = NemotronAsrStreamingEncoderCausalConv1d(
            channels,
            channels,
            kernel_size,
            cache_key=f"conv.{layer_idx}",
            stride=1,
            groups=channels,
            bias=config.convolution_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_masked_rows: torch.Tensor | None = None,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
    ):
        hidden_states = hidden_states.transpose(1, 2)  # (B, C, T)

        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        if all_masked_rows is not None:
            hidden_states = hidden_states.masked_fill(all_masked_rows, 0.0)

        hidden_states = self.depthwise_conv(hidden_states, padding_cache=padding_cache)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # (B, T, C)

        return hidden_states


class NemotronAsrStreamingEncoderAttention(ParakeetEncoderAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        total_key_length = key_states.shape[2]

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        query_states_with_bias_u = query_states + self.bias_u.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )
        query_states_with_bias_v = query_states + self.bias_v.view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(batch_size, -1, self.config.num_attention_heads, self.head_dim)

        matrix_bd = query_states_with_bias_v @ relative_key_states.permute(0, 2, 3, 1)
        matrix_bd = self._rel_shift(matrix_bd)
        matrix_bd = matrix_bd[..., :total_key_length]
        matrix_bd = matrix_bd * self.scaling

        if attention_mask is not None:
            matrix_bd = matrix_bd.masked_fill_(attention_mask.logical_not(), float("-inf"))

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states_with_bias_u,
            key=key_states,
            value=value_states,
            attention_mask=matrix_bd,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _mask_subsampled_frames(hidden_states: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Zero out time frames beyond each sequence's valid length so they don't leak into the next conv."""
    if lengths is None:
        return hidden_states
    time = torch.arange(hidden_states.shape[2], device=hidden_states.device)
    return hidden_states * (time < lengths[:, None])[:, None, :, None]


class NemotronAsrStreamingEncoderSubsamplingLayer(nn.Module):

    def __init__(self, config: NemotronAsrStreamingEncoderConfig, layer_idx: int):
        super().__init__()
        channels = config.subsampling_conv_channels
        self.depthwise_conv = NemotronAsrStreamingEncoderCausalConv2D(
            channels,
            channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
            groups=channels,
            cache_key=f"subsampling.{layer_idx}",
        )
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor | None,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.depthwise_conv(hidden_states, padding_cache=padding_cache)
        lengths = self.depthwise_conv.output_length(lengths, streaming=padding_cache is not None)
        hidden_states = self.pointwise_conv(hidden_states)
        return _mask_subsampled_frames(hidden_states, lengths), lengths


class NemotronAsrStreamingEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig):
        super().__init__()
        channels = config.subsampling_conv_channels
        num_layers = int(math.log2(config.subsampling_factor))

        self.conv_in = NemotronAsrStreamingEncoderCausalConv2D(
            1,
            channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
            cache_key="subsampling.0",
        )
        self.layers = nn.ModuleList(
            NemotronAsrStreamingEncoderSubsamplingLayer(config, layer_idx=i) for i in range(1, num_layers)
        )
        self.act_fn = nn.ReLU()
        self.linear = nn.Linear(config.subsampling_out_hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
    ):
        hidden_states = input_features.unsqueeze(1)
        lengths = attention_mask.sum(-1) if attention_mask is not None else None

        hidden_states = self.conv_in(hidden_states, padding_cache=padding_cache)
        lengths = self.conv_in.output_length(lengths, streaming=padding_cache is not None)
        hidden_states = self.act_fn(_mask_subsampled_frames(hidden_states, lengths))

        for layer in self.layers:
            hidden_states, lengths = layer(hidden_states, lengths, padding_cache=padding_cache)
            hidden_states = self.act_fn(hidden_states)

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[2], -1)
        hidden_states = self.linear(hidden_states)

        return hidden_states


class NemotronAsrStreamingEncoderBlock(ParakeetEncoderBlock):
    def __init__(self, config: NemotronAsrStreamingEncoderConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.conv = NemotronAsrStreamingEncoderConvolutionModule(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        all_masked_rows: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states  # the conformer architecture uses a factor of 0.5

        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output

        conv_output = self.conv(
            self.norm_conv(hidden_states), all_masked_rows=all_masked_rows, padding_cache=padding_cache
        )
        hidden_states = hidden_states + conv_output

        ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2_output  # the conformer architecture uses a factor of 0.5

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


@auto_docstring
class NemotronAsrStreamingPreTrainedModel(ParakeetPreTrainedModel):
    config: NemotronAsrStreamingConfig
    _supports_flex_attn = False

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        encoder_config = getattr(self.config, "encoder_config", self.config)

        kernel_size = encoder_config.subsampling_conv_kernel_size
        stride = encoder_config.subsampling_conv_stride
        num_layers = int(math.log2(encoder_config.subsampling_factor))

        all_paddings = (kernel_size - 1) + (stride - 1)
        add_pad = all_paddings - kernel_size
        lengths = input_lengths

        for _ in range(num_layers):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0
            lengths = torch.floor(lengths)

        return lengths.to(dtype=torch.int)


def chunked_limited_mask_function(left_ctx: int, right_ctx: int) -> Callable:
    """
    `chunked_limited` attention mask.
    """
    chunk_size = right_ctx + 1
    left_context_chunks = left_ctx // chunk_size if left_ctx >= 0 else 10_000

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        pass

    return inner_mask


@auto_docstring(
    custom_intro="""
    The NemotronAsrStreaming Encoder model, based on the [Fast Conformer architecture](https://huggingface.co/papers/2305.05084).
    """
)
class NemotronAsrStreamingEncoder(ParakeetEncoder):
    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attention_mask: bool = True,
        use_cache: bool | None = None,
        padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None,
        num_lookahead_tokens: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the output attention mask. Only effective when `attention_mask` is provided.
        past_key_values (`Cache`, *optional*):
            Sliding-window K/V cache (`DynamicCache` built from `config.sliding_window`) for cache-aware
            streaming attention.
        padding_cache (`NemotronAsrStreamingEncoderCausalConvPaddingCache`, *optional*):
            Unified streaming cache backing the subsampling Conv2d layers and the conformer depthwise Conv1d.
        num_lookahead_tokens (`int`, *optional*):
            Override of the right attention context (lookahead, in subsampled encoder frames) for this
            forward pass. Combined with the left context `config.sliding_window - 1`. Defaults to
            `config.default_num_lookahead_tokens`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, NemotronAsrStreamingEncoder
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> encoder = NemotronAsrStreamingEncoder.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> encoder_outputs = encoder(**inputs)

        >>> print(encoder_outputs.last_hidden_state.shape)
        ```
        """
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if padding_cache is None:
                padding_cache = NemotronAsrStreamingEncoderCausalConvPaddingCache()

        inputs_embeds = self.subsampling(input_features, attention_mask, padding_cache=padding_cache)
        inputs_embeds *= self.input_scale

        seq_length = inputs_embeds.shape[1]
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=seq_length)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=output_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            and_mask_function=chunked_limited_mask_function(*self._resolve_attn_context(num_lookahead_tokens)),
        )

        all_masked_rows = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                all_masked_rows = torch.all(~attention_mask, dim=-1)
            else:
                all_masked_rows = torch.all(attention_mask == 0.0, dim=-1)

        cached_frames = (
            past_key_values.get_mask_sizes(seq_length, 0)[0] - seq_length if past_key_values is not None else 0
        )
        position_embeddings = self.encode_positions(inputs_embeds, cached_frames=cached_frames)

        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.dropout_positions, training=self.training
        )

        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    all_masked_rows=all_masked_rows,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    padding_cache=padding_cache,
                    use_cache=use_cache,
                    **kwargs,
                )

        return NemotronAsrStreamingEncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_mask is not None and output_attention_mask else None,
            past_key_values=past_key_values,
            padding_cache=padding_cache,
        )

    def _resolve_attn_context(self, num_lookahead_tokens: int | None = None) -> tuple[int, int]:
        if num_lookahead_tokens is None:
            num_lookahead_tokens = self.config.default_num_lookahead_tokens
            logger.warning_once(
                f"`num_lookahead_tokens` was not provided. "
                f"Falling back to `config.default_num_lookahead_tokens={num_lookahead_tokens}`. "
                f"Consider preparing inputs with [`~NemotronAsrStreamingProcessor.__call__`] which automatically sets "
                f"this parameter."
            )

        left_context = self.config.sliding_window - 1
        return left_context, num_lookahead_tokens


@dataclass
class NemotronAsrStreamingRNNTOutput(ParakeetRNNTOutput):

    encoder_past_key_values: Cache | None = None
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None


class NemotronAsrStreamingRNNTDecoder(ParakeetRNNTDecoder):
    def __init__(self, config: NemotronAsrStreamingConfig):
        super().__init__(config)


class NemotronAsrStreamingRNNTJointNetwork(ParakeetRNNTJointNetwork):
    def __init__(self, config: NemotronAsrStreamingConfig):
        super().__init__(config)


@auto_docstring(
    custom_intro="""
    NemotronAsrStreaming Encoder with an RNN-T (Recurrent Neural Network Transducer) head.
    """
)
class NemotronAsrStreamingForRNNT(
    ParakeetForRNNT, NemotronAsrStreamingPreTrainedModel, NemotronAsrStreamingGenerationMixin
):
    config: NemotronAsrStreamingConfig

    def __init__(self, config: NemotronAsrStreamingConfig):
        super().__init__(config)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_cache: NemotronAsrStreamingRNNTDecoderCache | None = None,
        use_decoder_cache: bool | None = None,
        encoder_outputs: NemotronAsrStreamingEncoderModelOutput | None = None,
        labels: torch.Tensor | None = None,
        num_lookahead_tokens: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NemotronAsrStreamingRNNTOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Decoder input token ids for single-step inference.
        decoder_cache (`NemotronAsrStreamingRNNTDecoderCache`, *optional*):
            Decoder LSTM cache. Reused on blank predictions to skip the LSTM step.
        use_decoder_cache (`bool`, *optional*):
            Whether to allocate and use a decoder cache when none is provided.
        encoder_outputs (`NemotronAsrStreamingEncoderModelOutput`, *optional*):
            Pre-computed encoder outputs (last_hidden_state, pooler_output, ...).
        num_lookahead_tokens (`int`, *optional*):
            Right attention context (lookahead, in subsampled encoder frames) forwarded to the encoder.
            Defaults to `config.encoder_config.default_num_lookahead_tokens`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, NemotronAsrStreamingForRNNT
        >>> from datasets import load_dataset, Audio

        >>> model_id = "nvidia/nemotron-speech-streaming-en-0.6b"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = NemotronAsrStreamingForRNNT.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"])
        >>> outputs = model(**inputs)
        ```
        """
        if encoder_outputs is None:
            encoder_outputs = self.get_audio_features(
                input_features=input_features,
                attention_mask=attention_mask,
                num_lookahead_tokens=num_lookahead_tokens,
                **kwargs,
            )

        if use_decoder_cache and decoder_cache is None:
            decoder_cache = NemotronAsrStreamingRNNTDecoderCache()

        decoder_hidden_states = self.decoder(decoder_input_ids, cache=decoder_cache)
        logits = self.joint(
            encoder_hidden_states=encoder_outputs.pooler_output[:, :, None, :],
            decoder_hidden_states=decoder_hidden_states[:, None, :, :],
        ).squeeze(2)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, encoder_outputs=encoder_outputs)

        return NemotronAsrStreamingRNNTOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            decoder_cache=decoder_cache,
            encoder_past_key_values=encoder_outputs.past_key_values,
            padding_cache=encoder_outputs.padding_cache,
        )


__all__ = [
    "NemotronAsrStreamingConfig",
    "NemotronAsrStreamingEncoderConfig",
    "NemotronAsrStreamingFeatureExtractor",
    "NemotronAsrStreamingEncoderModelOutput",
    "NemotronAsrStreamingRNNTOutput",
    "NemotronAsrStreamingForRNNT",
    "NemotronAsrStreamingEncoder",
    "NemotronAsrStreamingPreTrainedModel",
]
