
import torch
from torch import nn

from ..core_model_loading import ConversionOps


class DecompressExperts(ConversionOps):

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        from compressed_tensors.compressors import BaseCompressor
        from compressed_tensors.compressors.format import infer_module_format

        ct_quantization_config = self.hf_quantizer.compressor.quantization_config

        quantization_scheme = list(ct_quantization_config.config_groups.values())[0]
        format = quantization_scheme.format or infer_module_format(nn.Linear, quantization_scheme)
        compressor = BaseCompressor.get_value_from_registry(format)

        class DummyModule(nn.Module):
            def __init__(self, weight, scale, shape):
                super().__init__()
                self.weight_packed = nn.Parameter(weight, requires_grad=False)
                self.weight_scale = nn.Parameter(scale, requires_grad=False)
                self.weight_shape = nn.Parameter(shape, requires_grad=False)

        pack_factor = 32 // quantization_scheme.weights.num_bits

        processed_out = {}
        for key, value in input_dict.items():
            if "weight_packed" not in key:
                continue
            quantized = value
            scales = input_dict[key.replace("weight_packed", "weight_scale")]
            shapes = input_dict.get(key.replace("weight_packed", "weight_shape"))

            output = None
            for i, (quant, scale) in enumerate(zip(quantized, scales)):
                stored_shape = None if shapes is None else shapes[i]
                if stored_shape is not None and stored_shape.numel():
                    shape = stored_shape
                else:
                    shape = torch.tensor([quant.shape[0], quant.shape[1] * pack_factor])
                module = DummyModule(quant, scale, shape)
                module.quantization_scheme = quantization_scheme
                compressor.decompress_module(module)

                if output is None:
                    output = torch.empty(
                        (len(quantized), *module.weight.shape),
                        dtype=module.weight.dtype,
                        device=module.weight.device,
                    )
                output[i].copy_(module.weight)
                del module

            del quantized, scales
            if output is not None:
                processed_out[key] = output

        return processed_out

    @property
    def reverse_op(self) -> "ConversionOps":
        pass
