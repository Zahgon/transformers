
import contextlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from functools import cached_property
from typing import Any, Literal

from .debug_utils import DebugOption
from .trainer_utils import (
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SaveStrategy,
    SchedulerType,
)
from .utils import (
    ACCELERATE_MIN_VERSION,
    ExplicitEnum,
    is_accelerate_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuron_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)
from .utils.generic import strtobool
from .utils.import_utils import enable_tf32, is_optimum_neuron_available


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

if is_torch_available():
    import torch
    import torch.distributed as dist

if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState
    from accelerate.utils import DistributedType

    from .trainer_pt_utils import AcceleratorConfig

if is_accelerate_available("1.10.1"):
    from accelerate.parallelism_config import ParallelismConfig
else:
    ParallelismConfig = Any

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_torch_neuroncore_available(check_device=False):
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        if is_optimum_neuron_available():
            logger.info(
                "Make sure that you are performing the training with the NeuronTrainer from optimum[neuron], this "
                "will fail otherwise."
            )
        else:
            logger.warning(
                "Please use the NeuronTrainer from optimum[neuron] instead of the Transformers library to perform "
                "training on AWS Trainium instances. More information here: "
                "https://github.com/huggingface/optimum-neuron"
            )
            import torch_xla.distributed.xla_backend as xbn

            if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                dist.init_process_group(backend="xla")
                if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                    raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


class OptimizerNames(ExplicitEnum):

    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    ADAMW_TORCH_4BIT = "adamw_torch_4bit"
    ADAMW_TORCH_8BIT = "adamw_torch_8bit"
    ADEMAMIX = "ademamix"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    ADEMAMIX_8BIT = "ademamix_8bit"
    LION_8BIT = "lion_8bit"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADEMAMIX = "paged_ademamix_32bit"
    PAGED_ADEMAMIX_8BIT = "paged_ademamix_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    RMSPROP = "rmsprop"
    RMSPROP_BNB = "rmsprop_bnb"
    RMSPROP_8BIT = "rmsprop_bnb_8bit"
    RMSPROP_32BIT = "rmsprop_bnb_32bit"
    GALORE_ADAMW = "galore_adamw"
    GALORE_ADAMW_8BIT = "galore_adamw_8bit"
    GALORE_ADAFACTOR = "galore_adafactor"
    GALORE_ADAMW_LAYERWISE = "galore_adamw_layerwise"
    GALORE_ADAMW_8BIT_LAYERWISE = "galore_adamw_8bit_layerwise"
    GALORE_ADAFACTOR_LAYERWISE = "galore_adafactor_layerwise"
    LOMO = "lomo"
    ADALOMO = "adalomo"
    GROKADAMW = "grokadamw"
    SCHEDULE_FREE_RADAM = "schedule_free_radam"
    SCHEDULE_FREE_ADAMW = "schedule_free_adamw"
    SCHEDULE_FREE_SGD = "schedule_free_sgd"
    APOLLO_ADAMW = "apollo_adamw"
    APOLLO_ADAMW_LAYERWISE = "apollo_adamw_layerwise"
    STABLE_ADAMW = "stable_adamw"


def _convert_str_dict(passed_value: dict):
    "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            if value.lower() in ("true", "false"):
                passed_value[key] = value.lower() == "true"
            elif value.isdigit():
                passed_value[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                passed_value[key] = float(value)

    return passed_value


@dataclass
class TrainingArguments:

    _VALID_DICT_FIELDS = [
        "accelerator_config",
        "fsdp_config",
        "deepspeed",
        "gradient_checkpointing_kwargs",
        "lr_scheduler_kwargs",
    ]

    output_dir: str | None = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    per_device_train_batch_size: int = field(default=8, metadata={"help": "The batch size per device for training."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "Overrides `num_train_epochs`. If set to a positive number, the total number of training steps to perform. Must be set when the training dataset does not implement `__len__` (e.g. a streaming dataset)."
        },
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for the optimizer."})
    lr_scheduler_type: SchedulerType | str = field(
        default="linear",
        metadata={"help": "The learning rate scheduler type to use. See `SchedulerType` for all possible values."},
    )
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "The extra arguments for the lr_scheduler. See the documentation of each scheduler for possible values."
        },
    )
    warmup_steps: float = field(
        default=0,
        metadata={
            "help": "Number of steps for a linear warmup from 0 to `learning_rate`. Can be an integer (exact steps) or a float in [0, 1) (ratio of total steps)."
        },
    )

    default_optim = "adamw_torch"
    if is_torch_available():
        from .pytorch_utils import is_torch_greater_or_equal_than_2_8

        if is_torch_greater_or_equal_than_2_8:
            default_optim = "adamw_torch_fused"
    optim: OptimizerNames | str = field(
        default=default_optim,
        metadata={"help": "The optimizer to use. See `OptimizerNames` for the complete list."},
    )
    optim_args: str | None = field(
        default=None,
        metadata={
            "help": "Optional arguments supplied to optimizers such as AnyPrecisionAdamW, AdEMAMix, and GaLore."
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={
            "help": "Weight decay coefficient applied by the optimizer. Automatically excluded from bias and LayerNorm parameters."
        },
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={
            "help": "The exponential decay rate for the first moment estimates (momentum) in Adam-based optimizers."
        },
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={
            "help": "The exponential decay rate for the second moment estimates (variance) in Adam-based optimizers."
        },
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon value for numerical stability in Adam-based optimizers."}
    )
    optim_target_modules: None | str | list[str] = field(
        default=None,
        metadata={"help": "The target modules to optimize. Currently used for the GaLore and APOLLO algorithms."},
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of update steps to accumulate gradients before performing a backward/update pass."
                " Effective batch size = per_device_train_batch_size * num_devices * gradient_accumulation_steps."
            )
        },
    )
    average_tokens_across_devices: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to average tokens across devices. If enabled, will use all_reduce to "
            "synchronize num_tokens_in_batch for precise loss calculation. Reference: "
            "https://github.com/huggingface/transformers/issues/34242"
        },
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Maximum gradient norm for gradient clipping. Set to 0 to disable."}
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "Label smoothing factor to prevent overconfidence. Zero means no smoothing."}
    )

    bf16: bool = field(
        default=False,
        metadata={
            "help": "Enable bfloat16 (BF16) mixed precision training. Generally preferred over FP16 due to better numerical stability."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Enable float16 (FP16) mixed precision training. Consider using BF16 instead if your hardware supports it."
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": "Use full BF16 precision for evaluation (not just mixed precision). Faster and saves memory."
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={
            "help": "Use full FP16 precision for evaluation (not just mixed precision). Faster and saves memory."
        },
    )
    tf32: bool | None = field(
        default=None,
        metadata={
            "help": "Enable TF32 mode on Ampere and newer GPUs. Provides up to 8x speedup with negligible accuracy loss."
        },
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training."
        },
    )
    gradient_checkpointing_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={"help": "Keyword arguments passed to `gradient_checkpointing_enable()`."},
    )

    torch_compile: bool = field(
        default=False, metadata={"help": "Compile the model using `torch.compile()` for faster training."}
    )
    torch_compile_backend: str | None = field(
        default=None,
        metadata={
            "help": "Backend for `torch.compile()`. If set, automatically enables `torch_compile`.",
        },
    )
    torch_compile_mode: str | None = field(
        default=None,
        metadata={
            "help": "Compilation mode for `torch.compile()`. If set, automatically enables `torch_compile`.",
        },
    )

    use_liger_kernel: bool = field(
        default=False,
        metadata={
            "help": "Enable Liger Kernel optimizations. Increases throughput by ~20%% and reduces memory by ~60%%."
        },
    )
    liger_kernel_config: dict[str, bool] | None = field(
        default=None,
        metadata={
            "help": "Configuration for Liger Kernel. Passed as kwargs to `_apply_liger_kernel_to_instance()`. If None, uses default configuration."
        },
    )

    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use cache for the model For training, this is usually not needed apart from some PEFT methods that uses `past_key_values`."
        },
    )
    neftune_noise_alpha: float | None = field(
        default=None,
        metadata={
            "help": "If not None, activates NEFTune noise embeddings. Can drastically improve performance for instruction fine-tuning. Typical range: [5.0, 15.0]."
        },
    )
    torch_empty_cache_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied."
        },
    )
    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": "Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors."
        },
    )

    logging_strategy: IntervalStrategy | str = field(
        default="steps",
        metadata={"help": "The logging strategy to adopt during training. Options: 'no', 'epoch', 'steps'."},
    )
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_first_step: bool = field(
        default=False, metadata={"help": "Whether to log the first `global_step` or not."}
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_nan_inf_filter: bool = field(
        default=True,
        metadata={
            "help": "Filter out NaN and Inf losses when logging. Does not affect gradient computation, only logging."
        },
    )
    include_num_input_tokens_seen: str | bool = field(
        default="no",
        metadata={
            "help": (
                "Whether to track the number of input tokens seen. "
                "Must be one of [`all`, `non_padding`, `no`] or a boolean value which map to `all` or `no`"
            )
        },
    )

    log_level: str = field(
        default="passive",
        metadata={
            "help": "Logging level for the main process. Options: 'debug', 'info', 'warning', 'error', 'critical', 'passive'.",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: str = field(
        default="warning",
        metadata={
            "help": "Logging level for replica processes in distributed training. Same options as `log_level`.",
            "choices": trainer_log_levels.keys(),
        },
    )
    disable_tqdm: bool | None = field(
        default=None,
        metadata={"help": "Disable tqdm progress bars. Defaults to True if log_level is warning or lower."},
    )

    report_to: None | str | list[str] = field(
        default="none",
        metadata={
            "help": "The list of integrations to report the results and logs to. Use 'all' for all installed integrations, 'none' for no integrations."
        },
    )
    run_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "An optional descriptor for the run. Notably used for trackio, wandb, mlflow comet and swanlab "
                "logging."
            )
        },
    )
    project: str = field(
        default="huggingface",
        metadata={"help": "The name of the project to use for logging. Currently, only used by Trackio."},
    )
    trackio_space_id: str | None = field(
        default=None,
        metadata={
            "help": (
                "Hugging Face Space id for live Gradio-based Trackio logging (read/write Bucket access). Use "
                "'username/reponame', 'orgname/reponame', or 'reponame' (current user's namespace). None: log only "
                "locally, no Space. Prefer trackio_static_space_id for stable post-training dashboard links. Public "
                "unless hub_private_repo=True or org default."
            )
        },
    )
    trackio_bucket_id: str | None = field(
        default=None,
        metadata={"help": "Optional HF Bucket id when using a Trackio Space; if unset, Trackio picks a default."},
    )
    trackio_static_space_id: str | None | Literal[False] = field(
        default=None,
        metadata={
            "help": (
                "Static read-only Trackio Space over the Bucket (stable model-card links). False: no static sync on Hub "
                "push and no freeze after training. None/str: allow static Space; for local-only logging, Hub push runs "
                "sync(static); after training, freeze runs only if trackio_space_id was set (Gradio Space). str sets "
                "explicit static Space id. Public unless hub_private_repo=True or org default."
            )
        },
    )

    eval_strategy: IntervalStrategy | str = field(
        default="no",
        metadata={"help": "When to run evaluation. Options: 'no', 'steps', 'epoch'."},
    )
    eval_steps: float | None = field(
        default=None,
        metadata={
            "help": (
                "Number of update steps between evaluations if `eval_strategy='steps'`. Defaults to `logging_steps` if not set."
                " Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    eval_delay: float = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " eval_strategy."
            )
        },
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "The batch size per device (GPU/TPU core/CPU) for evaluation."}
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and generating predictions, only returns the loss."},
    )
    eval_on_start: bool = field(
        default=False,
        metadata={
            "help": "Whether to run through the entire `evaluation` step at the very beginning of training as a sanity check."
        },
    )
    eval_do_concat_batches: bool = field(
        default=True,
        metadata={
            "help": "Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`, will instead store them as lists, with each batch kept separate."
        },
    )
    eval_use_gather_object: bool = field(
        default=False,
        metadata={
            "help": "Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices."
        },
    )
    eval_accumulation_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If unset, predictions are accumulated on the accelerator before being moved to the CPU."
        },
    )

    include_for_metrics: list[str] = field(
        default_factory=list,
        metadata={"help": "Include additional data in the `compute_metrics` function. Options: 'inputs', 'loss'."},
    )
    batch_eval_metrics: bool = field(
        default=False,
        metadata={"help": "Break eval metrics calculation into batches to save memory."},
    )

    save_only_model: bool = field(
        default=False,
        metadata={
            "help": "Save only model weights, not optimizer/scheduler/RNG state. Prevents resuming training from checkpoint."
        },
    )
    save_strategy: SaveStrategy | str = field(
        default="steps",
        metadata={
            "help": "The checkpoint save strategy to adopt during training. Options: 'no', 'epoch', 'steps', 'best'."
        },
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of checkpoints to keep. Deletes older checkpoints in `output_dir`. The best checkpoint is always retained when `load_best_model_at_end=True`."
        },
    )
    enable_jit_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Enable JIT checkpointing on SIGTERM signal for graceful termination on preemptible workloads. Configure your orchestrator's graceful shutdown period accordingly."
        },
    )

    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to push the model to the Hub every time the model is saved."}
    )
    hub_token: str | None = field(
        default=None,
        metadata={
            "help": "The token to use to push the model to the Hub. Defaults to the token from `hf auth login`."
        },
    )
    hub_private_repo: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to make the repo private. If `None` (default), the repo will be public unless the "
            "organization's default is private. This value is ignored if the repo already exists. If reporting to "
            "Trackio Spaces created or synced (including on Hub push when `trackio_space_id` is None) use the same "
            "logic for whether the Space is private."
        },
    )
    hub_model_id: str | None = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: HubStrategy | str = field(
        default="every_save",
        metadata={
            "help": "Defines what and when to push to Hub. Options: 'end', 'every_save', 'checkpoint', 'all_checkpoints'."
        },
    )
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
    )
    hub_revision: str | None = field(
        default=None,
        metadata={
            "help": "The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash."
        },
    )

    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Load the best checkpoint at the end of training. Requires `eval_strategy` to be set."},
    )
    metric_for_best_model: str | None = field(
        default=None,
        metadata={
            "help": "Metric to use for comparing models when `load_best_model_at_end=True`. Defaults to 'loss'."
        },
    )
    greater_is_better: bool | None = field(
        default=None,
        metadata={"help": "Whether higher metric values are better. Defaults based on `metric_for_best_model`."},
    )

    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, skip fast-forwarding through the dataset to reach the previous state. If True, training starts from the beginning of the dataset."
        },
    )
    restore_callback_states_from_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether to restore the callback states from the checkpoint. If `True`, will override callbacks passed to the `Trainer` if they exist in the checkpoint."
        },
    )

    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: int | None = field(
        default=None,
        metadata={"help": "Random seed to be used with data samplers. If not set, uses the same seed as `seed`."},
    )

    use_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use cpu. If set to False, we will use the available torch device/backend."
        },
    )

    accelerator_config: dict | str | None = field(
        default=None,
        metadata={
            "help": "Configuration for the internal Accelerate integration. Can be a path to a JSON config file or a dict."
        },
    )
    parallelism_config: ParallelismConfig | None = field(
        default=None,
        metadata={"help": "Parallelism configuration for the training run. Requires Accelerate `1.10.1`."},
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )
    dataloader_prefetch_factor: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
            )
        },
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Whether or not to automatically remove the columns unused by the model forward method."},
    )
    label_names: list[str] | None = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    train_sampling_strategy: str = field(
        default="random",
        metadata={
            "help": "Sampler for training: 'random' (default), 'sequential', or 'group_by_length'.",
            "choices": ["random", "sequential", "group_by_length"],
        },
    )
    length_column_name: str = field(
        default="length",
        metadata={
            "help": "Column name for precomputed lengths. Ignored unless `train_sampling_strategy` is 'group_by_length'."
        },
    )

    ddp_find_unused_parameters: bool | None = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_bucket_cap_mb: int | None = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_broadcast_buffers: bool | None = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `broadcast_buffers` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_static_graph: bool | None = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `static_graph` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_backend: str | None = field(
        default=None,
        metadata={
            "help": "The backend to use for distributed training. Must be one of 'nccl', 'mpi', 'xccl', 'gloo', 'hccl'.",
            "choices": ["nccl", "gloo", "mpi", "xccl", "hccl", "cncl", "mccl"],
        },
    )
    ddp_timeout: int = field(
        default=1800,
        metadata={"help": "The timeout for `torch.distributed.init_process_group` calls (in seconds)."},
    )

    fsdp: str | None = field(
        default=None,
        metadata={
            "help": "Enable PyTorch Fully Sharded Data Parallel (FSDP) for distributed training. Pass `--fsdp` (or `fsdp=True`) to turn FSDP on.",
            "nargs": "?",
            "const": True,
        },
    )
    fsdp_config: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Tuning for FSDP (used only when `fsdp` is enabled). Either a path to a JSON config file (e.g., `fsdp_config.json`) or an already loaded dict."
        },
    )

    deepspeed: dict | str | None = field(
        default=None,
        metadata={"help": "Enable DeepSpeed integration. Value is a path to a JSON config file or a dict."},
    )

    debug: str | list[DebugOption] = field(
        default="",
        metadata={
            "help": "Enable one or more debug features. Options: 'underflow_overflow' (detect overflow in model I/O), 'tpu_metrics_debug' (print TPU metrics)."
        },
    )
    skip_memory_metrics: bool = field(
        default=True,
        metadata={
            "help": "Whether to skip adding memory profiler reports to metrics. Skipped by default because it slows down training."
        },
    )

    do_train: bool = field(
        default=False,
        metadata={
            "help": "Whether to run training. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    do_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to run evaluation. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    do_predict: bool = field(
        default=False,
        metadata={
            "help": "Whether to run predictions on the test set. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={
            "help": "Path to a folder with a valid checkpoint for your model. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )

    local_rank: int = field(
        default=-1,
        metadata={
            "help": "When using torch.distributed.launch (Deprecated), it will pass `local_rank` in the script, so we need this for the parser. To get the local rank, prefer using the property `local_process_index`"
        },
    )

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            logger.info(
                "No output directory specified, defaulting to 'trainer_output'. "
                "To change this behavior, specify --output_dir when creating TrainingArguments."
            )

        for valid_field in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, valid_field)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, valid_field, loaded_dict)

        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.include_num_input_tokens_seen, bool):
            self.include_num_input_tokens_seen = "all" if self.include_num_input_tokens_seen else "no"

        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        self.optim = OptimizerNames(self.optim)

        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]
        elif self.debug is None:
            self.debug = []

        if self.do_eval is False and self.eval_strategy != IntervalStrategy.NO:
            self.do_eval = True

        if self.eval_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.eval_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        if (
            self.load_best_model_at_end
            or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
            or self.lr_scheduler_type == SchedulerType.GREEDY
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")

        if self.report_to == "all" or self.report_to == ["all"]:
            from .integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        from .integrations import is_kubeflow_available

        if is_kubeflow_available() and "kubeflow" not in self.report_to:
            self.report_to = list(self.report_to) + ["kubeflow"]

        self._validate_args()

        self.mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if self.fp16:
            self.mixed_precision = "fp16"
        elif self.bf16:
            self.mixed_precision = "bf16"

        if (self.torch_compile_mode is not None or self.torch_compile_backend is not None) and not self.torch_compile:
            self.torch_compile = True
        if self.torch_compile and self.torch_compile_backend is None:
            if not self.use_cpu and is_torch_hpu_available():
                self.torch_compile_backend = "hpu_backend"
            elif not self.use_cpu and is_torch_neuron_available():
                self.torch_compile_backend = "neuron"
            else:
                self.torch_compile_backend = "inductor"

        if self.torch_compile:
            if not is_accelerate_available("1.2.0"):
                os.environ["ACCELERATE_DYNAMO_BACKEND"] = self.torch_compile_backend
                if self.torch_compile_mode is not None:
                    os.environ["ACCELERATE_DYNAMO_MODE"] = self.torch_compile_mode

        if is_accelerate_available():
            if not isinstance(self.accelerator_config, AcceleratorConfig):
                if self.accelerator_config is None:
                    self.accelerator_config = AcceleratorConfig()
                elif isinstance(self.accelerator_config, dict):
                    self.accelerator_config = AcceleratorConfig(**self.accelerator_config)
                elif isinstance(self.accelerator_config, type):
                    raise NotImplementedError(
                        "Tried passing in a callable to `accelerator_config`, but this is not supported. "
                        "Please pass in a fully constructed `AcceleratorConfig` object instead."
                    )
                else:
                    self.accelerator_config = AcceleratorConfig.from_json_file(self.accelerator_config)
            if self.accelerator_config.split_batches:
                logger.info(
                    "Using `split_batches=True` in `accelerator_config` will override the `per_device_train_batch_size` "
                    "Batches will be split across all processes equally when using `split_batches=True`."
                )

        if is_torch_available():
            self.device

        if is_torch_available() and self.torch_compile:
            if is_torch_tf32_available():
                if self.tf32 is None and not self.fp16 or self.bf16:
                    device_str = "MUSA" if is_torch_musa_available() else "CUDA"
                    logger.info(
                        f"Setting TF32 in {device_str} backends to speedup torch compile, you won't see any improvement"
                        " otherwise."
                    )
                    enable_tf32(True)
            else:
                logger.warning(
                    "The speedups for torchdynamo mostly come with GPU Ampere or higher and which is not detected here."
                )
        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    enable_tf32(True)
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    enable_tf32(False)

        if self.use_cpu:
            self.dataloader_pin_memory = False

        self.fsdp_plugin_args = self._process_fsdp_args()

        self.deepspeed_plugin = None
        if self.deepspeed:
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

            from accelerate.utils import DeepSpeedPlugin

            self.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.hf_deepspeed_config)
        elif strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")):
            from accelerate.utils import DeepSpeedPlugin

            self.deepspeed_plugin = DeepSpeedPlugin()
            self.deepspeed_plugin.set_mixed_precision(self.mixed_precision)
            self.deepspeed_plugin.set_deepspeed_weakref()

    def _validate_args(self):
        """Validate argument combinations and value constraints."""
        if self.torch_empty_cache_steps is not None:
            if not (isinstance(self.torch_empty_cache_steps, int) and self.torch_empty_cache_steps > 0):
                raise ValueError(
                    f"`torch_empty_cache_steps` must be an integer bigger than 0, got {self.torch_empty_cache_steps}."
                )

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.eval_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == SaveStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        if self.load_best_model_at_end and self.save_strategy != SaveStrategy.BEST:
            if self.eval_strategy != self.save_strategy:
                raise ValueError(
                    '--load_best_model_at_end requires the save and eval strategy to match, except when --save_strategy="best", but found\n- Evaluation '
                    f"strategy: {self.eval_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.eval_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps "
                            f"{self.save_steps} and eval_steps {self.eval_steps}."
                        )
                    LARGE_MULTIPLIER = 1_000_000
                    if (self.save_steps * LARGE_MULTIPLIER) % (self.eval_steps * LARGE_MULTIPLIER) != 0:
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            f"steps, but found {self.save_steps}, which is not a multiple of {self.eval_steps}."
                        )
                else:
                    raise ValueError(
                        "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                        f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                    )

        if is_torch_available():
            if self.bf16 or self.bf16_full_eval:
                if not self.use_cpu and not is_torch_bf16_gpu_available() and not is_torch_xla_available():
                    error_message = "Your setup doesn't support bf16/gpu. You need to assign use_cpu if you want to train the model on CPU."
                    if is_torch_cuda_available():
                        error_message += " You need Ampere+ GPU with cuda>=11.0."
                    raise ValueError(error_message)

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.eval_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires an eval strategy")
            if not is_torch_available():
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires torch>=0.2.0")

        if self.lr_scheduler_type == SchedulerType.GREEDY:
            if self.eval_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type greedy requires an eval strategy")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be an integer or a float")

        if self.dataloader_num_workers == 0 and self.dataloader_prefetch_factor is not None:
            raise ValueError(
                "--dataloader_prefetch_factor can only be set when data is loaded in a different process, i.e."
                " when --dataloader_num_workers > 0."
            )

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        pass

    @property
    def eval_batch_size(self) -> int:
        pass

    @property
    def ddp_timeout_delta(self) -> timedelta:
        pass

    @cached_property
    def _setup_devices(self) -> "torch.device":
        pass

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        requires_backends(self, ["torch"])
        return self._setup_devices

    @property
    def n_gpu(self):
        pass

    @property
    def parallel_mode(self):
        pass

    @property
    def world_size(self):
        pass

    @property
    def process_index(self):
        pass

    @property
    def local_process_index(self):
        pass

    @property
    def should_log(self):
        pass

    @property
    def should_save(self):
        pass

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
        anything) unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """

        log_level = trainer_log_levels[self.log_level]
        log_level_replica = trainer_log_levels[self.log_level_replica]

        log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
        log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
        return log_level_main_node if self.should_log else log_level_replica_node

    @property
    def place_model_on_device(self) -> bool | None:
        pass

    @property
    def _no_sync_in_gradient_accumulation(self):
        pass

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        pass

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            int(self.warmup_steps) if self.warmup_steps >= 1 else math.ceil(num_training_steps * self.warmup_steps)
        )
        return warmup_steps

    def _dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None and not isinstance(d["dtype"], str):
            d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self._dict_dtype_to_str(value)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
            if is_accelerate_available() and isinstance(v, AcceleratorConfig):
                d[k] = v.to_dict()
            if k == "model_init_kwargs" and isinstance(v, dict) and "quantization_config" in v:
                quantization_config = v.get("quantization_config")
                if quantization_config and not isinstance(quantization_config, dict):
                    d[k]["quantization_config"] = quantization_config.to_dict()
            if k == "parallelism_config" and v is not None:
                d[k] = v.to_json()

        self._dict_dtype_to_str(d)

        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> dict[str, Any]:
        pass

    def set_training(
        self,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        weight_decay: float = 0,
        num_epochs: float = 3,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        gradient_checkpointing: bool = False,
    ):
        pass

    def set_evaluate(
        self,
        strategy: str | IntervalStrategy = "no",
        steps: int = 500,
        batch_size: int = 8,
        accumulation_steps: int | None = None,
        delay: float | None = None,
        loss_only: bool = False,
    ):
        pass

    def set_testing(
        self,
        batch_size: int = 8,
        loss_only: bool = False,
    ):
        pass

    def set_save(
        self,
        strategy: str | IntervalStrategy = "steps",
        steps: int = 500,
        total_limit: int | None = None,
        on_each_node: bool = False,
    ):
        pass

    def set_logging(
        self,
        strategy: str | IntervalStrategy = "steps",
        steps: int = 500,
        report_to: str | list[str] = "none",
        level: str = "passive",
        first_step: bool = False,
        nan_inf_filter: bool = False,
        on_each_node: bool = False,
        replica_level: str = "passive",
    ):
        pass

    def set_push_to_hub(
        self,
        model_id: str,
        strategy: str | HubStrategy = "every_save",
        token: str | None = None,
        private_repo: bool | None = None,
        always_push: bool = False,
        revision: str | None = None,
    ):
        pass

    def set_optimizer(
        self,
        name: str | OptimizerNames = "adamw_torch",
        learning_rate: float = 5e-5,
        weight_decay: float = 0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        args: str | None = None,
    ):
        pass

    def set_lr_scheduler(
        self,
        name: str | SchedulerType = "linear",
        num_epochs: float = 3.0,
        max_steps: int = -1,
        warmup_steps: float = 0,
    ):
        pass

    def set_dataloader(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        auto_find_batch_size: bool = False,
        ignore_data_skip: bool = False,
        sampler_seed: int | None = None,
    ):
        pass

    def _process_fsdp_args(self):
        if not self.fsdp:
            return None

        if self.fsdp_config is None:
            self.fsdp_config = {}
        elif isinstance(self.fsdp_config, str):
            with open(self.fsdp_config, encoding="utf-8") as f:
                self.fsdp_config = json.load(f)
        for k in list(self.fsdp_config):
            if k.startswith("fsdp_"):
                self.fsdp_config[k[5:]] = self.fsdp_config.pop(k)

        if isinstance(self.fsdp, (str, list)):
            self._apply_legacy_fsdp_to_config(self.fsdp, self.fsdp_config)
        self.fsdp = True

        if self.gradient_checkpointing:
            logger.warning(
                "When using FSDP, prefer `activation_checkpointing` in `fsdp_config` over "
                "`gradient_checkpointing`; the latter introduces a redundant AllGather in the backward pass. "
                "Reference: https://github.com/huggingface/transformers/issues/30404"
            )

        self.fsdp_config["min_num_params"] = self.fsdp_config.get("min_num_params", 0)
        if isinstance(self.fsdp_config.get("transformer_layer_cls_to_wrap"), str):
            self.fsdp_config["transformer_layer_cls_to_wrap"] = [self.fsdp_config["transformer_layer_cls_to_wrap"]]
        self.fsdp_config.setdefault("xla", False)
        self.fsdp_config.setdefault("xla_fsdp_v2", False)
        self.fsdp_config.setdefault("xla_fsdp_grad_ckpt", False)

        if self.fsdp_config["xla"]:
            self.xla_fsdp_config = self.fsdp_config.get("xla_fsdp_settings", {}).copy()
            if "compute_dtype" in self.xla_fsdp_config:
                self.xla_fsdp_config["compute_dtype"] = getattr(torch, self.xla_fsdp_config["compute_dtype"])
            if "buffer_dtype" in self.xla_fsdp_config:
                self.xla_fsdp_config["buffer_dtype"] = getattr(torch, self.xla_fsdp_config["buffer_dtype"])
            return None
        elif self.fsdp_config["xla_fsdp_grad_ckpt"]:
            warnings.warn("`--xla_fsdp_grad_ckpt` is useful only when `--xla` is set to true.")

        from accelerate.utils.constants import FSDP_AUTO_WRAP_POLICY

        fsdp_version = int(self.fsdp_config.get("version", 2))
        fsdp_plugin_args = {"fsdp_version": fsdp_version}

        if self.fsdp_config.get("cpu_offload", False):
            fsdp_plugin_args["cpu_offload"] = True

        auto_wrap_policy = self.fsdp_config.get("auto_wrap_policy", FSDP_AUTO_WRAP_POLICY[0])
        if auto_wrap_policy not in FSDP_AUTO_WRAP_POLICY:
            raise ValueError(f"`auto_wrap_policy` must be one of {FSDP_AUTO_WRAP_POLICY}, got {auto_wrap_policy}.")
        fsdp_plugin_args["auto_wrap_policy"] = auto_wrap_policy
        if auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[1] and self.fsdp_config["min_num_params"] > 0:
            fsdp_plugin_args["min_num_params"] = self.fsdp_config["min_num_params"]
        elif (
            auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[0]
            and self.fsdp_config.get("transformer_layer_cls_to_wrap") is not None
        ):
            fsdp_plugin_args["transformer_cls_names_to_wrap"] = ",".join(
                self.fsdp_config["transformer_layer_cls_to_wrap"]
            )

        cpu_ram_efficient_loading = str(self.fsdp_config.get("cpu_ram_efficient_loading", "false")).lower()
        fsdp_plugin_args["cpu_ram_efficient_loading"] = str_to_bool(cpu_ram_efficient_loading)
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = cpu_ram_efficient_loading

        fsdp_plugin_args["state_dict_type"] = self.fsdp_config.get("state_dict_type", "FULL_STATE_DICT")

        if "activation_checkpointing" in self.fsdp_config:
            fsdp_plugin_args["activation_checkpointing"] = str_to_bool(
                str(self.fsdp_config["activation_checkpointing"]).lower()
            )

        if fsdp_version == 2:
            fsdp_plugin_args["reshard_after_forward"] = str_to_bool(
                str(self.fsdp_config.get("reshard_after_forward", True)).lower()
            )
        else:
            logger.warning(
                "FSDP1 (`fsdp_config['version'] = 1`) is deprecated and will be removed in Transformers "
                "v5.20. Please migrate to FSDP2 by setting `fsdp_config['version'] = 2` (the default)."
            )
            fsdp_plugin_args["reshard_after_forward"] = str(
                self.fsdp_config.get("reshard_after_forward", "full_shard")
            ).lower()
            fsdp_plugin_args["forward_prefetch"] = str_to_bool(
                str(self.fsdp_config.get("forward_prefetch", "false")).lower()
            )
            fsdp_plugin_args["backward_prefetch"] = self.fsdp_config.get("backward_prefetch", "NO_PREFETCH").upper()
            fsdp_plugin_args["use_orig_params"] = str_to_bool(
                str(self.fsdp_config.get("use_orig_params", "true")).lower()
            )
            fsdp_plugin_args["sync_module_states"] = str_to_bool(
                str(self.fsdp_config.get("sync_module_states", "true")).lower()
            )
            if "limit_all_gathers" in self.fsdp_config:
                fsdp_plugin_args["limit_all_gathers"] = str_to_bool(str(self.fsdp_config["limit_all_gathers"]).lower())

        return fsdp_plugin_args

    @staticmethod
    def _apply_legacy_fsdp_to_config(fsdp, fsdp_config):
        """
        Translate legacy `fsdp` values (string / list of [`~trainer_utils.FSDPOption`]) into
        `fsdp_config` entries, using the shape expected by the target FSDP version:

        - `"offload"` → `fsdp_config["cpu_offload"] = True`
        - Sharding strategies → `fsdp_config["reshard_after_forward"]`. For FSDP2 this is a
          bool (`"full_shard"` → `True`, `"shard_grad_op"` → `False`); for FSDP1 it is the
          lowercase strategy name (`"full_shard"`, `"hybrid_shard"`, ...).
        - `"auto_wrap"` → no-op (default `auto_wrap_policy` already wraps).

        Isolated so the deprecated path can be removed in one place once support is dropped.
        """
        if isinstance(fsdp, str):
            logger.warning(
                "Passing `fsdp` as a string is deprecated and will be removed in Transformers v5.20. "
                "Use `fsdp=True` and configure everything via `fsdp_config` instead."
            )
            items = fsdp.split()
        else:
            logger.warning(
                "Passing `fsdp` as a list is deprecated and will be removed in Transformers v5.20. "
                "Use `fsdp=True` and configure everything via `fsdp_config` instead."
            )
            items = list(fsdp)

        from accelerate.utils.constants import FSDP_SHARDING_STRATEGY

        version = int(fsdp_config.get("version", 2))
        for item in items:
            if item.upper() in FSDP_SHARDING_STRATEGY:
                if version == 2:
                    if item == FSDPOption.FULL_SHARD:
                        fsdp_config.setdefault("reshard_after_forward", True)
                    elif item == FSDPOption.SHARD_GRAD_OP:
                        fsdp_config.setdefault("reshard_after_forward", False)
                    else:
                        raise ValueError(
                            f"`fsdp={item}` is only available with FSDP1. Set `fsdp_config['version'] = 1` to "
                            f"use it, but note that FSDP1 is deprecated and will be removed in Transformers v5.20."
                        )
                else:
                    fsdp_config.setdefault("reshard_after_forward", item)
            elif item == FSDPOption.OFFLOAD:
                fsdp_config.setdefault("cpu_offload", True)
            elif item == FSDPOption.AUTO_WRAP:
                pass
            else:
                raise ValueError(f"Unknown `fsdp` option: {item}")


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
    SAGEMAKER_MODEL_PARALLEL = "sagemaker_model_parallel"
    SAGEMAKER_DATA_PARALLEL = "sagemaker_data_parallel"
    TPU = "tpu"


def str_to_bool(value, to_bool: bool = True) -> int | bool:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1 if not to_bool else True
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0 if not to_bool else False
    else:
        raise ValueError(f"invalid truth value {value}")
