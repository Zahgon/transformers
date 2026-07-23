
import logging

import torch

from ..cache_utils import (
    DynamicCache,
    DynamicLayer,
    DynamicSlidingWindowLayer,
    EncoderDecoderCache,
    StaticCache,
    StaticLayer,
    StaticSlidingWindowLayer,
)
from ..generation.configuration_utils import GenerationConfig
from ..modeling_utils import PreTrainedModel
from ..pytorch_utils import (
    is_torch_greater_or_equal,
    is_torch_greater_or_equal_than_2_6,
)


class TorchExportableModuleForVLM:

    def __init__(self, model, max_batch_size: int = 1, max_cache_len: int = 1024):
        """
        Initialize the exportable VLM module.

        Args:
            model: The VLM (e.g. SmolVLM) model instance
            max_batch_size: Maximum batch size. Always 1 for ExecuTorch
            max_cache_len: Maximum cache length for text generation
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.config = model.config

        self.vision_encoder = model.model.vision_model
        self.connector = model.model.connector
        self.text_decoder = model.model.text_model

        self.exported_vision_encoder = None
        self.exported_connector = None
        self.exported_text_decoder = None

    def export_vision_encoder(self):
        pass

    def export_connector(self):
        pass

    def export_text_decoder(self):
        pass

    def export(self, **kwargs):
        pass

    def forward(self, pixel_values, input_ids, cache_position):
        """
        Simplified forward pass for inference with guaranteed non-null input_ids and cache_position.

        Args:
            pixel_values: Input images [1, channels, height, width] (optional)
            input_ids: Text token IDs [1, seq_len] (required - won't be None)
            cache_position: Cache positions [seq_len] (required - won't be None)

        Returns:
            Output with logits for text generation
        """

    def generate(
        self, pixel_values=None, input_ids=None, max_new_tokens=50, do_sample=False, temperature=1.0, **kwargs
    ):
        """
        Simplified generate method with guaranteed non-null input_ids.

        Args:
            pixel_values: Input images [1, channels, height, width] (optional)
            input_ids: Initial text tokens [1, seq_len] (required - won't be None)
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling or greedy decoding
            temperature: Temperature for sampling

        Returns:
            Generated sequences
        """


class TorchExportableModuleForDecoderOnlyLM(torch.nn.Module):

    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int | None = None,
        max_cache_len: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Initializes the exportable module.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.

        Raises:
            ValueError: If the model is configured with a unsupported cache implementation.
        """
        super().__init__()

        config = model.config.get_text_config()

        if not hasattr(config, "use_cache") or config.use_cache is False:
            raise ValueError("The model must have caching enabled to be performant.")

        if hasattr(config, "layer_types") and getattr(config, "sliding_window", None) is not None:
            self.model = TorchExportableModuleWithHybridCache(model, batch_size, max_cache_len, device)
        else:
            logging.info(
                "Using `StaticCache` for export as `layer_types` is not specified or `sliding_window` is `null` in the config."
            )
            self.model = TorchExportableModuleWithStaticCache(model, batch_size, max_cache_len, device)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            inputs_embeds (`torch.Tensor`): Tensor representing current input embeddings to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        return self.model.forward(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def export(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        dynamic_shapes: dict | None = None,
        strict: bool | None = None,
    ) -> torch.export.ExportedProgram:
        pass

    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        device: str = "cpu",
    ) -> str:
        """
        Generate a sequence of tokens using an exported program.

        Args:
            exported_program (`torch.export.ExportedProgram`): The exported model being used for generate.
            tokenizer: The tokenizer to use.
            prompt (str): The input prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling or greedy decoding.
            temperature (float): The temperature for sampling.
            top_k (int): The number of highest probability tokens to keep for top-k sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            device (str): The device to use.

        Returns:
            str: The generated text.
        """
        exported_module = exported_program.module()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generated_ids = input_ids.clone()

        curr_position = 0
        for i in range(input_ids.shape[1]):
            curr_input_ids = input_ids[:, i : i + 1]
            curr_cache_position = torch.tensor([curr_position], dtype=torch.long, device=device)

            _ = exported_module(input_ids=curr_input_ids, cache_position=curr_cache_position)
            curr_position += 1

        for _ in range(max_new_tokens):
            curr_input_ids = generated_ids[:, -1:]
            curr_cache_position = torch.tensor([curr_position], dtype=torch.long, device=device)

            outputs = exported_module(input_ids=curr_input_ids, cache_position=curr_cache_position)

            if do_sample:
                if temperature > 0:
                    logits = outputs / temperature
                else:
                    logits = outputs

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token_id = outputs.argmax(dim=-1, keepdim=True)

            if next_token_id.dim() > 2:
                next_token_id = next_token_id.squeeze(-1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            curr_position += 1

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def get_head_shapes(config) -> tuple[int | list[int], int | list[int]]:
    """Returns a tuple `(num_heads, head_dim)` containing either 2 ints, or a list of int with the value for each
    layer."""
    if hasattr(config, "global_head_dim"):
        head_dim = [
            config.global_head_dim if layer == "full_attention" else config.head_dim
            for layer in config.layer_types[: -config.num_kv_shared_layers]
        ]
        num_heads = [
            config.num_global_key_value_heads
            if layer == "full_attention" and config.attention_k_eq_v
            else config.num_key_value_heads
            for layer in config.layer_types[: -config.num_kv_shared_layers]
        ]
    else:
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    return num_heads, head_dim


class TorchExportableModuleWithStaticCache(torch.nn.Module):

    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int | None = None,
        max_cache_len: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Initializes the wrapper module with the pretrained model.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap. The model must have caching
                enabled and use a 'static' caching implementation.
            batch_size (`Optional[int]`): The batch size of the model. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we raise a ValueError.
            max_cache_len (`Optional[int]`): The maximum cache length for generation. Same mechanism as `batch_size` if
                not provided.
            device (`Optional[torch.device]`): The device to use. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we use `model.device` (no error is raised).

        Raises:
            AssertionError: If the pretrained model does not have caching enabled or if it does
            not use a 'static' caching implementation in `model.generation_config`.
            ValueError: If `batch_size` or `max_cache_len` is not provided, either as an argument or in `cache_config`.
        """
        super().__init__()

        config = model.config.get_text_config()
        generation_config = model.generation_config

        if generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config` in `model`."
            )
        if not generation_config.use_cache:
            raise AssertionError(
                "The model must have caching enabled to be exported with static caching. "
                "Please set `generation_config.use_cache=True`."
            )
        if generation_config.cache_implementation != "static":
            raise AssertionError(
                "The model must use a 'static' caching implementation to be exported with static caching. "
                "Please set `generation_config.cache_implementation='static'`."
            )

        cache_config = {} if generation_config.cache_config is None else generation_config.cache_config

        if batch_size is None:
            batch_size = cache_config.get("batch_size", None)
            if batch_size is None:
                raise ValueError("batch_size must be provided, either as an argument or in cache_config.")
        if max_cache_len is None:
            max_cache_len = cache_config.get("max_cache_len", None)
            if max_cache_len is None:
                raise ValueError("max_cache_len must be provided, either as an argument or in cache_config.")
        if device is None:
            device = cache_config.get("device", model.device)

        self.model = model
        self.static_cache = StaticCache(max_cache_len=max_cache_len, config=config)
        for i, layer in enumerate(self.static_cache.layers):
            if isinstance(layer, StaticSlidingWindowLayer):
                self.static_cache.layers[i] = StaticLayer(max_cache_len)
        num_heads, head_dim = get_head_shapes(config)
        dtype = self.model.dtype
        self.static_cache.early_initialization(batch_size, num_heads, head_dim, dtype, device)

        for i, layer in enumerate(self.static_cache.layers):
            self.register_buffer(f"key_cache_{i}", layer.keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", layer.values, persistent=False)
            self.register_buffer(f"cumulative_length_{i}", layer.cumulative_length, persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            inputs_embeds (`torch.Tensor`): Tensor representing current input embeddings to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.

        This forward adapter serves two primary purposes:

        1. **Making the Model `torch.export`-Compatible**:
            The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
            enabling the model to be exportable using `torch.export` without encountering issues.

        2. **Ensuring Compatibility with `ExecuTorch` runtime**:
            The adapter matches the model's forward signature with that in `executorch/extension/llm/runner`,
            ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.
        """
        for layer in self.static_cache.layers:
            layer.cumulative_length.copy_(cache_position[0])

        past_key_values = self.static_cache

        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if hasattr(outs, "logits"):
            return outs.logits
        else:
            return outs.last_hidden_state

    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens using an exported program.

        This util function is designed to test exported models by simulating the generation process.
        It processes the input prompt tokens sequentially (no parallel prefill).
        This generate function is not intended to replace the original `generate` method, and the support
        for leveraging the original `generate` is potentially planned!

        Args:
            exported_program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
            prompt_token_ids (`torch.Tensor`): Tensor representing the input prompt token IDs.
            max_new_tokens (`int`): Maximum number of new tokens to generate. Note that the total generation
                length is limited by both `max_new_tokens` and the model's cache size.

        Returns:
            torch.Tensor: A tensor containing the generated sequence of token IDs, including the original prompt tokens.
        """
        device = prompt_token_ids.device
        prompt_token_len = prompt_token_ids.shape[-1]
        max_generation_length = prompt_token_len + max_new_tokens
        for buffer_name, buffer in exported_program.named_buffers():
            if buffer_name.startswith("key_cache"):
                max_cache_len = buffer.shape[2]
                max_generation_length = min(max_generation_length, max_cache_len)
                break

        response_tokens = []
        for input_pos in range(min(max_generation_length, prompt_token_len)):
            result = exported_program.module().forward(
                input_ids=prompt_token_ids[:, input_pos : input_pos + 1],
                cache_position=torch.tensor([input_pos], dtype=torch.long, device=device),
            )
            response_tokens.append(prompt_token_ids[0][input_pos].item())

        current_token = torch.argmax(result[:, -1, :], dim=-1).item()
        response_tokens.append(current_token)

        while len(response_tokens) < max_generation_length:
            result = exported_program.module().forward(
                input_ids=torch.tensor([[current_token]], dtype=torch.long, device=device),
                cache_position=torch.tensor([len(response_tokens)], dtype=torch.long, device=device),
            )
            current_token = torch.argmax(result[:, -1, :], dim=-1).item()
            response_tokens.append(current_token)

        return torch.tensor([response_tokens], dtype=torch.long, device=device)


class TorchExportableModuleWithHybridCache(torch.nn.Module):

    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int | None = None,
        max_cache_len: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Initializes the exportable module.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.
            batch_size (`Optional[int]`): The batch size of the model. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we raise a ValueError.
            max_cache_len (`Optional[int]`): The maximum cache length for generation. Same mechanism as `batch_size` if
                not provided.
            device (`Optional[torch.device]`): The device to use. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we use `model.device` (no error is raised).
        Raises:
            AssertionError: If the model doesn't have the expected configuration for hybrid StaticCache.
            ValueError: If `batch_size` or `max_cache_len` is not provided, either as an argument or in `cache_config`.
        """
        super().__init__()
        self.model = model
        config = model.config.get_text_config()
        generation_config = model.generation_config

        if generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config` in `model`."
            )
        if not config.use_cache:
            raise AssertionError("Model must have caching enabled.")

        cache_config = {} if generation_config.cache_config is None else generation_config.cache_config
        if batch_size is None:
            batch_size = cache_config.get("batch_size", None)
            if batch_size is None:
                raise ValueError("batch_size must be provided, either as an argument or in cache_config.")
        if max_cache_len is None:
            max_cache_len = cache_config.get("max_cache_len", None)
            if max_cache_len is None:
                raise ValueError("max_cache_len must be provided, either as an argument or in cache_config.")
        if device is None:
            device = cache_config.get("device", model.device)

        self.cache = StaticCache(config=config, max_cache_len=max_cache_len)
        for i, layer in enumerate(self.cache.layers):
            if isinstance(layer, StaticSlidingWindowLayer):
                self.cache.layers[i] = StaticLayer(max_cache_len)
        num_heads, head_dim = get_head_shapes(config)
        dtype = self.model.dtype
        self.cache.early_initialization(batch_size, num_heads, head_dim, dtype, device)

        for i, layer in enumerate(self.cache.layers):
            self.register_buffer(f"key_cache_{i}", layer.keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", layer.values, persistent=False)
            self.register_buffer(f"cumulative_length_{i}", layer.cumulative_length, persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            inputs_embeds (`Optional[torch.Tensor]`): Tensor representing current input embeddings to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        for layer in self.cache.layers:
            layer.cumulative_length.copy_(cache_position[0])

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=self.cache,
            use_cache=True,
        )

        return outputs.logits


def convert_and_export_with_cache(
    model: PreTrainedModel,
    example_input_ids: torch.Tensor | None = None,
    example_cache_position: torch.Tensor | None = None,
    dynamic_shapes: dict | None = None,
    strict: bool | None = None,
):
    pass


class Seq2SeqLMEncoderExportableModule(torch.nn.Module):

    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids).last_hidden_state


class Seq2SeqLMDecoderExportableModuleWithStaticCache(torch.nn.Module):

    def __init__(self, model, max_static_cache_length, batch_size):
        super().__init__()

        self.decoder = model.get_decoder()
        self.lm_head = model.lm_head
        self.config = model.config

        model_device = next(model.parameters()).device

        self.static_cache = StaticCache(config=self.config, max_cache_len=max_static_cache_length)
        for i, layer in enumerate(self.static_cache.layers):
            if isinstance(layer, StaticSlidingWindowLayer):
                self.static_cache.layers[i] = StaticLayer(max_static_cache_length)
        num_heads, head_dim = get_head_shapes(self.config)
        self.static_cache.early_initialization(batch_size, num_heads, head_dim, torch.float32, model_device)
        self.cache = EncoderDecoderCache(self.static_cache, DynamicCache(config=self.config))

        register_dynamic_cache_export_support()

        for i, layer in enumerate(self.static_cache.layers):
            self.register_buffer(f"key_cache_{i}", layer.keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", layer.values, persistent=False)
            self.register_buffer(f"cumulative_length_{i}", layer.cumulative_length, persistent=False)

    def forward(self, decoder_input_ids, encoder_hidden_states, cache_position):
        for layer in self.static_cache.layers:
            layer.cumulative_length.copy_(cache_position[0])

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=self.cache,
            use_cache=True,
        )

        lm_logits = self.lm_head(outputs[0])

        return lm_logits


class Seq2SeqLMExportableModule(torch.nn.Module):
    def __init__(
        self, model, batch_size=1, max_hidden_seq_length=4096, cache_implementation="static", max_cache_length=1024
    ):
        super().__init__()

        self.full_model = model
        self.encoder = model.get_encoder()
        self.config = model.config
        self.max_hidden_seq_length = max_hidden_seq_length
        self.generation_config = GenerationConfig(
            use_cache=True,
            max_length=max_cache_length,
            cache_implementation=cache_implementation,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_cache_length,
            },
            eos_token_id=model.generation_config.eos_token_id,
        )
        self.exported_encoder = None
        self.exported_decoder = None

    def _export_encoder(self, encoder_input_ids):
        pass

    def _export_decoder(self, decoder_input_ids, encoder_hidden_states, cache_position):
        pass

    def export(self, encoder_input_ids=None, decoder_input_ids=None, encoder_hidden_states=None, cache_position=None):
        pass

    def generate(self, prompt_token_ids, max_new_tokens):
        with torch.no_grad():
            model_device = self.full_model.device

            if prompt_token_ids.device != model_device:
                prompt_token_ids = prompt_token_ids.to(model_device)

            encoder_output = self.exported_encoder.module()(prompt_token_ids)

            decoder_input_ids = torch.tensor([[0]], dtype=torch.long, device=model_device)
            generated_ids = [0]

            for i in range(max_new_tokens - 1):
                logits = self.exported_decoder.module()(
                    decoder_input_ids, encoder_output, torch.tensor([i], dtype=torch.long, device=model_device)
                )

                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_token)

                decoder_input_ids = torch.tensor([[next_token]], dtype=torch.long, device=model_device)

                if next_token == self.generation_config.eos_token_id:
                    break

            return generated_ids


def export_with_dynamic_cache(
    model: PreTrainedModel,
    example_input_ids: torch.Tensor | None = None,
    example_attention_mask: torch.Tensor | None = None,
):
    pass


def register_dynamic_cache_export_support():
    """
    Utilities for `DynamicCache` <> torch.export support
    """

    try:
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_cache_dict(dynamic_cache)),
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
                _get_cache_dict(dynamic_cache)
            ),
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_cache_dict(cache), spec),
        )
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def _get_cache_dict(cache: DynamicCache):
    """Convert cache to dictionary format for pytree operations."""
    if any(not isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)) for layer in cache.layers):
        raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

    if not is_torch_greater_or_equal_than_2_6:
        logging.warning("DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions.")

    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [layer.values for layer in cache.layers if layer.values is not None],
    }


def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
    pass
