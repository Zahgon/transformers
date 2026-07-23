
from ..utils import is_compressed_tensors_available, is_torch_available, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter
    from ..integrations.compressed_tensors import DecompressExperts


logger = logging.get_logger(__name__)


class CompressedTensorsHfQuantizer(HfQuantizer):

    requires_calibration = True
    quantization_config: CompressedTensorsConfig

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)


        quantization_config.post_init()
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)
        self.run_compressed = quantization_config.run_compressed
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires compressed-tensors>=0.15.0: "
                "`pip install compressed-tensors>=0.15.0`"
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype != torch.float16:
            logger.info("We suggest you to set `dtype=torch.float16` for better efficiency with compressed_tensors.")
        return dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        apply_quantization_config(model, ct_quantization_config, self.run_compressed)
        if self.quantization_config.is_quantization_compressed:
            self.compressor.compress_model(model=model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Decompress loaded model if necessary - need for qat"""

        if self.quantization_config.is_quantization_compressed and not self.run_compressed:
            self.compressor.decompress_model(model=model)

    def update_tp_plan(self, config):
        additional_plan = {
            "layers.*.feed_forward.experts.*.gate_proj.weight": "colwise",
            "layers.*.feed_forward.experts.*.gate_proj.weight_scale": "colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight": "colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight_scale": "colwise",
            "layers.*.feed_forward.experts.*.down_proj.weight": "rowwise",
        }
        if config.get_text_config() is not None and config.get_text_config().base_model_tp_plan is not None:
            config.get_text_config().base_model_tp_plan.update(additional_plan)

        return config

    @property
    def is_trainable(self):
        pass

    def is_qat_trainable(self) -> bool:
        pass

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True

    def get_weight_conversions(self):
        dequant_conversions = [
            WeightConverter(
                source_patterns=[
                    r".weight_packed$",
                    r".weight_scale$",
                    r".weight_shape$",
                ],
                target_patterns=r"weight",
                operations=[DecompressExperts(self)],
            ),
        ]
        return dequant_conversions

    def update_weight_conversions(self, weight_conversions):
        updated: list = []
        for conv in weight_conversions:
            if not isinstance(conv, WeightConverter) or any("experts" not in p for p in conv.source_patterns):
                updated.append(conv)
                continue
            weight_sources = [p for p in conv.source_patterns if p.endswith(".weight")]
            if weight_sources:
                packed_weight = [p + "_packed$" for p in weight_sources]
                scale_sources = [p + "_scale$" for p in weight_sources]
                shape_sources = [p + "_shape$" for p in weight_sources]
                other = [p for p in conv.source_patterns if not p.endswith(".weight")]
                new_sources = packed_weight + scale_sources + shape_sources + other
                new_ops = [DecompressExperts(self)] + list(conv.operations)
                conv = WeightConverter(
                    source_patterns=new_sources,
                    target_patterns=conv._original_target_patterns,
                    operations=new_ops,
                )
            updated.append(conv)
        updated.extend(self.get_weight_conversions())
        return updated
