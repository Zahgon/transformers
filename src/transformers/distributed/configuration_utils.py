
import json
import os
from dataclasses import asdict, dataclass

from ..utils import is_torch_available


if is_torch_available():
    import torch


@dataclass
class DistributedConfig:

    tp_size: int | None = None
    tp_plan: dict[str, str] | None = None
    enable_sequence_parallel: bool = False
    enable_expert_parallel: bool = False
    fsdp_size: int | None = None
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = False

    def __post_init__(self):
        if self.tp_size is None and self.fsdp_size is None:
            return

        if self.tp_size is None:
            self.tp_size = 1
        if self.fsdp_size is None:
            self.fsdp_size = 1

        if self.tp_size > 1 and self.fsdp_size > 1:
            raise ValueError(
                "FSDP+TP is not supported yet. "
                "Use DistributedConfig(fsdp_size=N) or DistributedConfig(tp_size=N), not both. "
                "2D support will come soon."
            )

    def validate(self) -> None:
        """Validate against the live process group. Call before distributed load/train."""
        if self.tp_size is None and self.fsdp_size is None:
            return

        if self.tp_size <= 1 and self.fsdp_size <= 1:
            return

        if not is_torch_available():
            raise RuntimeError("PyTorch is required to use DistributedConfig.")

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before using DistributedConfig with tp_size > 1 or "
                "fsdp_size > 1. Call dist.init_process_group(...) first, or launch with torchrun."
            )

        world_size = torch.distributed.get_world_size()
        if self.tp_size * self.fsdp_size != world_size:
            raise RuntimeError(
                f"tp_size ({self.tp_size}) * fsdp_size ({self.fsdp_size}) is not equal to world_size ({world_size})"
            )

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "DistributedConfig":
        merged = {**config_dict, **kwargs}
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in merged.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
