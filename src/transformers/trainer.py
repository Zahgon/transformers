
import contextlib
import functools
import glob
import inspect
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any


from .integrations import (
    get_reporting_integration_callbacks,
)


import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from huggingface_hub import CommitInfo, ModelCard
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from . import __version__
from .configuration_utils import PreTrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .distributed.fsdp import get_fsdp_ckpt_kwargs, update_fsdp_plugin_peft
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .feature_extraction_utils import FeatureExtractionMixin
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .image_processing_utils import BaseImageProcessor
from .integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    deepspeed_sp_compute_loss,
    is_deepspeed_available,
    propagate_args_to_deepspeed,
)
from .integrations.liger import apply_liger_kernel
from .integrations.neftune import activate_neftune, deactivate_neftune
from .integrations.peft import MIN_PEFT_VERSION
from .integrations.tpu import save_tpu_checkpoint, tpu_spmd_dataloader, wrap_model_xla_fsdp
from .loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, unwrap_model
from .models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from .optimization import GreedyLR, get_scheduler
from .processing_utils import ProcessorMixin
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_optimizer import (
    _OPTIMIZER_HANDLERS,
    OptimizerContext,
    _parse_optim_args,
    is_optimizer_factory,
)
from .trainer_pt_utils import (
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    distributed_broadcast_scalars,
    find_batch_size,
    get_model_param_count,
    get_parameter_names,
    is_attention_mask_causal,
    nested_detach,
    nested_gather,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    safe_globals,
    set_rng_state_for_device,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    _is_peft_model,
    align_special_tokens,
    compare_trainer_and_checkpoint_args,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    load_sharded_checkpoint,
    number_of_arguments,
    rotate_checkpoints,
    seed_worker,
    set_seed,
    sort_checkpoints,
    speed_metrics,
    suppress_progress_bars,
    unwrap_peft_model,
    validate_quantization_for_training,
)
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    can_return_loss,
    check_torch_load_is_safe,
    find_labels,
    hf_api,
    is_accelerate_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    logging,
)
from .utils.import_utils import requires
from .utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_nested_concat

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        DataLoaderConfiguration,
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        release_memory,
        save_fsdp_model,
        save_fsdp_optimizer,
    )
    from accelerate.utils.memory import clear_device_cache

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


@requires(
    backends=(
        "torch",
        "accelerate",
    )
)
class Trainer:

    from .trainer_pt_utils import (
        get_learning_rates,
        get_num_trainable_parameters,
        get_optimizer_group,
        log_metrics,
        metrics_format,
        save_metrics,
        save_state,
    )


    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: "Dataset | IterableDataset | datasets.Dataset | None" = None,
        eval_dataset: "Dataset | dict[str, Dataset] | datasets.Dataset | None" = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[..., PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        self.deepspeed = None
        self.model = model
        self.create_accelerator_and_postprocess()

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        args._setup_devices  # force device and distributed setup init explicitly

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                raise ValueError("`Trainer` requires either a `model` or `model_init` argument, but not both.")
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        validate_quantization_for_training(model)

        self.is_model_parallel = False
        if getattr(model, "hf_device_map", None) is not None:
            devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])

        self.is_fsdp_xla_enabled = args.fsdp and args.fsdp_config.get("xla", False)
        if args.fsdp:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if not self.is_fsdp_xla_enabled and args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise ValueError("Using fsdp only works in distributed training.")

        if args.place_model_on_device is not None:
            self.place_model_on_device = args.place_model_on_device
        elif (
            self.is_model_parallel
            or self.is_deepspeed_enabled
            or (args.fp16_full_eval or args.bf16_full_eval)
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
            or is_sagemaker_mp_enabled()
        ):
            self.place_model_on_device = False
        else:
            self.place_model_on_device = True

        if (
            self.place_model_on_device
            and getattr(model, "quantization_method", None) != QuantizationMethod.BITS_AND_BYTES
        ):
            self._move_model_to_device(model, args.device)

        if self.is_model_parallel:
            self.args._n_gpu = 1

        self.model_wrapped = model
        self.model = model

        unwrapped_model = unwrap_peft_model(self.accelerator.unwrap_model(model))

        if hasattr(unwrapped_model, "accepts_loss_kwargs"):
            self.model_accepts_loss_kwargs = unwrapped_model.accepts_loss_kwargs
        else:
            forward_params = inspect.signature(unwrapped_model.forward).parameters
            self.model_accepts_loss_kwargs = any(
                k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values()
            )

        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
            self.model_accepts_loss_kwargs = False

        model_to_inspect = unwrap_peft_model(self.model)
        default_label_names = find_labels(model_to_inspect.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(model_to_inspect.__class__)

        self._loss_shifts_labels = LOSS_MAPPING.get(getattr(model_to_inspect, "loss_type", None)) is ForCausalLMLoss

        if self.args.label_smoothing_factor != 0:
            if getattr(self.model.config, "problem_type", None) == "multi_label_classification":
                warnings.warn(
                    "Label smoothing is not compatible with multi-label classification. "
                    "Disabling label smoothing for this training run.",
                    UserWarning,
                )
                self.label_smoother = None
            else:
                self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        default_collator = (
            DataCollatorWithPadding(processing_class)
            if processing_class is not None
            and isinstance(processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
            else default_data_collator
        )
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs

        self._validate_args()

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)

        if self.args.enable_jit_checkpoint:
            from .trainer_jit_checkpoint import JITCheckpointCallback

            jit_callback = JITCheckpointCallback()
            default_callbacks = default_callbacks + [jit_callback]
            jit_callback.set_trainer(self)

        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        self.hub_model_id = None  # Set by init_hf_repo() when push_to_hub is enabled
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.control = TrainerControl()
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.is_in_train = False  # True between train() entry and exit
        self.hp_name = None  # Set by hyperparameter_search() to label the trial
        self.hp_search_backend = None  # Set by hyperparameter_search() (optuna / ray / wandb)
        self.current_flos = 0
        self._loggers_initialized = False
        self._signature_columns = None
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = self.args.use_cache

        self.is_fsdp_xla_v2_enabled = args.fsdp and args.fsdp_config.get("xla_fsdp_v2", False)
        if self.is_fsdp_xla_v2_enabled:
            if not IS_XLA_FSDPV2_POST_2_2:
                raise ValueError("FSDPv2 requires `torch_xla` 2.2 or higher.")
            num_devices = xr.global_runtime_device_count()
            xs.set_global_mesh(xs.Mesh(np.array(range(num_devices)), (num_devices, 1), axis_names=("fsdp", "tensor")))
        self.is_fsdp_xla_v1_enabled = self.is_fsdp_xla_enabled and not self.is_fsdp_xla_v2_enabled

        self._memory_tracker.stop_and_update_metrics()

    def _validate_args(self) -> None:
        """Validate constructor arguments and fail fast on incompatible combinations."""
        args = self.args

        if is_sagemaker_mp_enabled():
            if args.bf16:
                raise ValueError("SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead ")
            if args.fp16 != smp.state.cfg.fp16:
                logger.warning(
                    f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                    f"but FP16 provided in trainer argument is {args.fp16}, "
                    f"setting to {smp.state.cfg.fp16}"
                )
                args.fp16 = smp.state.cfg.fp16

        if args.batch_eval_metrics and self.compute_metrics is not None:
            if "compute_result" not in inspect.signature(self.compute_metrics).parameters:
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        if args.eval_strategy is not None and args.eval_strategy != "no" and self.eval_dataset is None:
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy} but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. "
            )
        if args.save_strategy == SaveStrategy.BEST or args.load_best_model_at_end:
            if args.metric_for_best_model is None:
                raise ValueError(
                    "`args.metric_for_best_model` must be provided when using 'best' save_strategy or if `args.load_best_model_at_end` is set to `True`."
                )

        if self.optimizer_cls_and_kwargs is not None and self.optimizer is not None:
            raise RuntimeError("Passing both `optimizers` and `optimizer_cls_and_kwargs` arguments is incompatible.")
        if self.model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_xla_available() and self.optimizer is not None:
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )
        if (self.is_fsdp_xla_enabled or self.is_fsdp_enabled) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise TypeError("The `data_collator` should be a simple callable (function, class with `__call__`).")
        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")
        if self.train_dataset is not None and not has_length(self.train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, so the number of training steps cannot be inferred; "
                "max_steps has to be specified. The total number of steps must be known in advance to bound the "
                "training loop and to configure the learning rate scheduler."
            )

        if self.train_dataset is not None and isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            logger.info(
                f"The `train_sampling_strategy='{args.train_sampling_strategy}'` option is ignored when using an `IterableDataset`. "
                "Samplers cannot be used with IterableDataset as they require indexed access to the dataset."
            )

    def _build_accelerator_args(self, **kwargs) -> dict[str, Any]:
        """Helper method to build accelerator-specific keyword arguments."""
        args = {
            "mixed_precision": self.args.mixed_precision,
            "deepspeed_plugin": self.args.deepspeed_plugin,
        }
        args.update(kwargs)

        if self.args.ddp_find_unused_parameters is not None:
            find_unused = self.args.ddp_find_unused_parameters
        elif isinstance(self.model, PreTrainedModel):
            find_unused = not (self.model.is_gradient_checkpointing or self.args.gradient_checkpointing)
        else:
            find_unused = True

        ddp_kwargs = {"find_unused_parameters": find_unused}
        if self.args.ddp_bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
        if self.args.ddp_broadcast_buffers is not None:
            ddp_kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers
        if self.args.ddp_static_graph is not None:
            ddp_kwargs["static_graph"] = self.args.ddp_static_graph

        args["kwargs_handlers"] = [DistributedDataParallelKwargs(**ddp_kwargs)]

        if self.args.parallelism_config is not None:
            min_accelerate_version = "1.12.0"
            if not is_accelerate_available(min_accelerate_version):
                raise ImportError(
                    f"ParallelismConfig requires accelerate>={min_accelerate_version}). Please upgrade accelerate to use this feature."
                )
            args["parallelism_config"] = self.args.parallelism_config

        if getattr(self.model, "tp_size", None) is not None and self.model.tp_size > 1:
            if self.args.parallelism_config is None:
                if is_accelerate_available("1.12.0"):
                    if self.args.parallelism_config is None:
                        from accelerate import ParallelismConfig

                        args["parallelism_config"] = ParallelismConfig(tp_size=self.model.tp_size)
                else:
                    raise ValueError("Requires accelerate>1.12.0 to use Tensor Parallelism.")
            elif args["parallelism_config"].tp_size != self.model.tp_size:
                args["parallelism_config"].tp_size = self.model.tp_size

        if is_accelerate_available("1.2.0"):
            from accelerate.utils import TorchDynamoPlugin

            dynamo_plugin = TorchDynamoPlugin(
                backend=self.args.torch_compile_backend, mode=self.args.torch_compile_mode
            )
            args["dynamo_plugin"] = dynamo_plugin

        return args

    def create_accelerator_and_postprocess(self) -> None:
        """Create the accelerator and perform post-creation setup (FSDP, DeepSpeed, etc.)."""
        grad_acc_kwargs = {}
        if self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        grad_acc_kwargs["num_steps"] = 1

        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_config = self.args.accelerator_config.to_dict()

        dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
        dataloader_config = DataLoaderConfiguration(
            **{param: accelerator_config.pop(param) for param in dataloader_params}
        )
        dataloader_config.data_seed = self.args.data_seed

        non_blocking = accelerator_config.pop("non_blocking")

        if non_blocking and not self.args.dataloader_pin_memory:
            logger.warning(
                "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
            )
        dataloader_config.non_blocking = non_blocking
        accelerator_config.pop("gradient_accumulation_kwargs")

        fsdp_plugin = None
        if self.args.fsdp_plugin_args is not None:
            from accelerate.utils import FullyShardedDataParallelPlugin

            fsdp_plugin = FullyShardedDataParallelPlugin(**self.args.fsdp_plugin_args)

        args = self._build_accelerator_args(
            dataloader_config=dataloader_config,
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        self.accelerator = Accelerator(**args)
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters:
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        if self.is_fsdp_enabled:
            if self.accelerator.state.fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            propagate_args_to_deepspeed(self.accelerator, self.args)

        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        if (
            self.is_deepspeed_enabled
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP"
            )
        if (
            self.args.save_only_model
            and self.is_fsdp_enabled
            and "SHARDED_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)
        ):
            raise ValueError("save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'")


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        return self._get_dataloader(
            dataset=eval_dataset,
            description="Evaluation",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            dataloader_key=dataloader_key,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed.
        """
        return self._get_dataloader(
            dataset=test_dataset,
            description="test",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
        )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Callable[[Dataset], torch.utils.data.Sampler] | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> torch.utils.data.Sampler | None:
        pass

    def _get_eval_sampler(self, eval_dataset: Dataset) -> torch.utils.data.Sampler | None:
        pass

    def _set_signature_columns_if_needed(self) -> None:
        """Populate `_signature_columns` from the model's forward signature if not already set."""
        if self._signature_columns is None:
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    model_to_inspect = self.model.base_model.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", description: str | None = None
    ) -> "datasets.Dataset":
        """Remove dataset columns not accepted by the model's forward method."""
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(self, data_collator: Callable, description: str | None = None) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator


    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        pass

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.

        Returns:
            `torch.optim.Optimizer`: The optimizer instance.
        """
        opt_model = self.model if model is None else model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if is_optimizer_factory(optimizer_cls):
                self.optimizer = optimizer_cls()(opt_model, **optimizer_kwargs)
            else:
                if "params" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("params")

                if "model" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("model")

                if "optimizer_dict" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.

        Returns:
            `torch.optim.lr_scheduler.LRScheduler`: The learning rate scheduler instance.
        """
        if self.lr_scheduler is None:
            if optimizer is None:
                if is_sagemaker_mp_enabled() and smp.state.cfg.fp16:
                    optimizer = self.optimizer.optimizer
                else:
                    optimizer = self.optimizer
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments, model: PreTrainedModel | None = None) -> tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.
            model (`PreTrainedModel`, *optional*):
                The model being trained. Required for some optimizers (GaLore, Apollo, LOMO).

        Returns:
            A tuple containing the optimizer class and a dictionary of optimizer keyword arguments.
        """
        ctx = OptimizerContext(
            args=args,
            model=model,
            optimizer_kwargs={"lr": args.learning_rate},
            adam_kwargs={
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
            },
            optim_args=_parse_optim_args(args.optim_args),
        )

        handler = _OPTIMIZER_HANDLERS.get(args.optim)
        if handler is None:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")

        return handler(ctx)

    def get_decay_parameter_names(self, model: nn.Module) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def _get_learning_rate(self) -> float:
        pass


    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: "optuna.Trial | dict[str, Any] | None" = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`list[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.

        Returns:
            [`~trainer_utils.TrainOutput`]: Object containing the global step count, training loss, and metrics.
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        if self.model_init is not None:
            enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)
            self.model = self.call_model_init(trial)
            self.optimizer, self.lr_scheduler = None, None
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        if self.args.use_liger_kernel:
            apply_liger_kernel(self.model, self.args.liger_kernel_config)

        if (args.fp16_full_eval or args.bf16_full_eval) and not self.is_model_parallel and self.model_init is None:
            self._move_model_to_device(self.model, args.device)

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        if isinstance(self.processing_class, (PreTrainedTokenizerBase, ProcessorMixin)) and hasattr(
            self.model, "config"
        ):
            align_special_tokens(self.model, self.processing_class)

        if self.neftune_noise_alpha is not None:
            self.neftune_hook_handle = activate_neftune(self.model, self.neftune_noise_alpha, self.accelerator)

        self._hp_search_setup(trial)

        if DebugOption.UNDERFLOW_OVERFLOW in args.debug:
            if args.n_gpu > 1:
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP with torchrun"
                )
            else:
                DebugUnderflowOverflow(self.model)

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None and args.auto_find_batch_size:
                self._train_batch_size = state.train_batch_size

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        ctx = suppress_progress_bars() if args.push_to_hub else contextlib.nullcontext()
        with ctx:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self,
        batch_size: int | None = None,
        args: TrainingArguments | None = None,
        resume_from_checkpoint: str | None = None,
        trial: "optuna.Trial | dict[str, Any] | None" = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        pass

    def _init_training_state(
        self, max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint, trial
    ) -> tuple[int, int]:
        pass

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        pass

    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        pass

    def _finalize_training(self, trial, num_train_samples, start_time):
        pass

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        pass

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, torch.Tensor | Any]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If not passed, the loss is computed
                using the default batch size reduction logic.

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculation might be slightly inaccurate when performing gradient accumulation.
        """
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled and self.model.training:
            return deepspeed_sp_compute_loss(self.accelerator, model, inputs, return_outputs, pc)

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)

        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch=num_items_in_batch,
            )
        elif labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = (
                unwrapped_model.base_model.model._get_name()
                if _is_peft_model(unwrapped_model)
                else unwrapped_model._get_name()
            )
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True, num_items_in_batch=num_items_in_batch)
            else:
                loss = self.label_smoother(outputs, labels, num_items_in_batch=num_items_in_batch)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss_scale = self.accelerator.num_processes
            if (pc := getattr(self.accelerator, "parallelism_config", None)) is not None:
                loss_scale //= pc.tp_size
            loss *= loss_scale if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss

    def compute_loss_context_manager(self) -> contextlib.ExitStack:
        """
        A helper wrapper to group together context managers.
        """
        ctx_stack = contextlib.ExitStack()

        autocast_ctx = self.autocast_smart_context_manager()
        if not isinstance(autocast_ctx, contextlib.nullcontext):
            ctx_stack.enter_context(autocast_ctx)

        return ctx_stack

    def autocast_smart_context_manager(self, cache_enabled: bool | None = True) -> contextlib.AbstractContextManager:
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation. We rely on accelerate for autocast, hence we do nothing here.
        """
        return contextlib.nullcontext()

    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: nn.Module,
        trial: "optuna.Trial | dict[str, Any] | None",
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        pass

    def get_batch_samples(
        self, epoch_iterator: Iterator, num_batches: int, device: torch.device
    ) -> tuple[list, torch.Tensor | int | None]:
        pass

    def _get_num_items_in_batch(self, batch_samples: list, device: torch.device) -> torch.Tensor | int | None:
        """
        Counts the number of items in the batches to properly scale the loss.
        Args:
            batch_samples (`list`): List of batches
            device (`torch.device`): The device on which the number of items in the batch should be.
        Returns:
            None if the number of items in the batch doesn't need to be computed else the number of items in the batch
        """
        num_items_in_batch = None
        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and (
                self.model_accepts_loss_kwargs
                or self.compute_loss_func is not None
            )
        )
        if count_num_items_in_batch:
            try:
                labels_for_count = [
                    batch["shift_labels"]
                    if "shift_labels" in batch
                    else batch["labels"][..., 1:]
                    if self._loss_shifts_labels
                    else batch["labels"]
                    for batch in batch_samples
                ]
                num_items_in_batch = sum(labels.ne(-100).sum() for labels in labels_for_count)
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                if self.args.world_size > 1:
                    num_items_in_batch = self.accelerator.gather(num_items_in_batch.to(device)).sum()
            elif self.args.n_gpu > 1:
                num_items_in_batch = num_items_in_batch // self.args.n_gpu

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    num_items_in_batch = num_items_in_batch.unsqueeze(0).expand(self.args.n_gpu, -1)
                if pc := getattr(self.accelerator, "parallelism_config", None):
                    num_items_in_batch = num_items_in_batch // pc.non_data_parallel_size

        return num_items_in_batch

    def _prepare_input(self, data: torch.Tensor | Any) -> torch.Tensor | Any:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )

        return inputs

    def _prepare_context_parallel_inputs(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[Callable, dict[str, torch.Tensor | Any]]:
        pass

    def set_initial_training_values(
        self, args: TrainingArguments, dataloader: DataLoader
    ) -> tuple[int, int, int, int, int, int | None, int]:
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `total_train_batch_size`
        - `steps_in_epoch` (total batches per epoch)
        - `max_steps`
        """
        max_steps = args.max_steps
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None
        total_train_batch_size = self.get_total_train_batch_size(args)

        sp_size = self.get_sp_size()
        if sp_size > 1 and len_dataloader is not None:
            len_dataloader = len_dataloader * sp_size

        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        if len_dataloader:
            num_examples = self.num_examples(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        steps_in_epoch = len_dataloader if len_dataloader is not None else max_steps * args.gradient_accumulation_steps
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        )

    def get_total_train_batch_size(self, args: TrainingArguments) -> int:
        """Calculates total batch size (micro_batch * grad_accum * dp_world_size).

        Accounts for all parallelism dimensions: TP, CP, and SP.

        Formula: dp_world_size = world_size // (tp_size * cp_size * sp_size)

        Where:
        - TP (Tensor Parallelism): Model layers split across GPUs
        - CP (Context Parallelism): Sequences split using Ring Attention (FSDP2)
        - SP (Sequence Parallelism): Sequences split using ALST/Ulysses (DeepSpeed)

        All dimensions are separate and multiplicative: world_size = dp_size * tp_size * cp_size * sp_size
        """

        dp_world_size = args.world_size // self.get_tp_size() // self.get_cp_size() // self.get_sp_size()
        return self._train_batch_size * args.gradient_accumulation_steps * dp_world_size

    def get_sp_size(self) -> int:
        """Get the sequence parallel size"""
        if getattr(self.accelerator, "parallelism_config", None) is None:
            return 1
        else:
            pc = self.accelerator.parallelism_config
            return pc.sp_size

    def get_cp_size(self) -> int:
        """Get the context parallel size"""
        if getattr(self.accelerator, "parallelism_config", None) is None:
            return 1
        else:
            pc = self.accelerator.parallelism_config
            return pc.cp_size

    def get_tp_size(self) -> int:
        """Get the tensor parallel size from either the model or DeepSpeed config."""

        if (model_tp := getattr(self.model, "_tp_size", None)) is not None:
            return model_tp

        if self.is_deepspeed_enabled and (deepspeed_config := getattr(self.args, "hf_deepspeed_config", None)):
            return deepspeed_config.config.get("tensor_parallel", {}).get("autotp_size", 1)

        return 1

    def _wrap_model(self, model: nn.Module, training: bool = True, dataloader: DataLoader | None = None) -> nn.Module:
        """Wrap `model` for distributed training if needed (DDP, FSDP, SageMaker, etc.)."""
        if self.accelerator.unwrap_model(model, keep_torch_compile=False) is not model:
            return model

        if is_sagemaker_mp_enabled():
            if isinstance(model, smp.model.DistributedModel):
                return model
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        if (
            self.args.n_gpu > 1
            and not getattr(model, "is_loaded_in_8bit", False)
            and not getattr(model, "is_loaded_in_4bit", False)
        ):
            model = nn.DataParallel(model)

        if not training:
            return model

        if self.is_fsdp_xla_enabled:
            model = wrap_model_xla_fsdp(model, self.args, self.is_fsdp_xla_v2_enabled)
        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        return model

    def _update_auto_batch_size(self, batch_size):
        pass

    def _track_num_input_tokens(self, inputs):
        pass

    def _clip_grad_norm(self, model):
        pass

    def _get_grad_norm(self, model, grad_norm=None):
        pass


    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset` | dict[str, `Dataset`], *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        self.gather_function = self.accelerator.gather_for_metrics

        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(
        self, test_dataset: Dataset, ignore_keys: list[str] | None = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss")
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                        if self.args.use_liger_kernel and prediction_loss_only:
                            inputs = {**inputs, "skip_logits": True}
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                        )
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _evaluate(
        self,
        trial: "optuna.Trial | dict[str, Any] | None",
        ignore_keys_for_eval: list[str] | None,
        skip_scheduler: bool = False,
    ) -> dict[str, float]:
        pass


    def _get_output_dir(self, trial: "optuna.Trial | dict[str, Any] | None") -> str:
        pass

    def _save_checkpoint(self, model: nn.Module, trial: "optuna.Trial | dict[str, Any] | None") -> None:
        pass

    def _determine_best_metric(self, metrics: dict[str, float], trial: "optuna.Trial | dict[str, Any] | None") -> bool:
        pass

    def _save_rng_state(self, output_dir: str) -> None:
        pass

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        pass

    def _save_scaler(self, output_dir: str) -> None:
        pass


    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: nn.Module | None = None) -> None:
        """Load model weights from a checkpoint directory."""
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PreTrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            if is_sagemaker_mp_enabled():
                smp.resume_from_checkpoint(
                    path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                )
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **get_fsdp_ckpt_kwargs(),
                )
            else:
                if os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    check_torch_load_is_safe()
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

                load_result = model.load_state_dict(state_dict, False)
                del state_dict
                self._issue_warnings_after_load(load_result)

        elif _is_peft_model(model):
            if hasattr(model, "active_adapters") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint):
                    active_adapters = model.active_adapters
                    if len(active_adapters) > 1:
                        logger.warning("Multiple active adapters detected will only consider the first adapter")
                    active_adapter = active_adapters[0]

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning(f"Could not load adapter model, make sure to have PEFT >= {MIN_PEFT_VERSION} installed")
        else:
            load_result = load_sharded_checkpoint(model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled())
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _load_best_model(self) -> None:
        pass

    def _load_rng_state(self, checkpoint: str | None) -> None:
        pass

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        pass

    def _load_scaler(self, checkpoint: str | None) -> None:
        pass

    def _load_callback_state(self) -> None:
        pass

    def _issue_warnings_after_load(self, load_result: Any) -> None:
        """Log warnings for missing or unexpected keys after loading a checkpoint."""
        if len(load_result.missing_keys) != 0:
            if (
                self.model._keys_to_ignore_on_save is not None
                and len(self.model._keys_to_ignore_on_save) > 0
                and set(load_result.missing_keys) == set(self.model._keys_to_ignore_on_save)
            ):
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )


    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_xla_available():
            save_tpu_checkpoint(
                self.model, self.args, self.accelerator, self.processing_class, self.is_fsdp_xla_v1_enabled, output_dir
            )
        elif is_sagemaker_mp_enabled():
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.is_fsdp_enabled:
            if "FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            try:
                accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                    inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
                )
                zero3_sharding = self.deepspeed.config.get("zero_optimization", {}).get("stage", None) == 3
                if accept_exclude_frozen_parameters and _is_peft_model(self.model) and zero3_sharding:
                    state_dict = self.deepspeed._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=True)
                else:
                    state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save", revision=self.args.hub_revision)

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        """Save model weights, configuration, and processing class to `output_dir`."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model, keep_torch_compile=False), supported_classes):
                self.accelerator.unwrap_model(self.model, keep_torch_compile=False).save_pretrained(
                    output_dir, state_dict=state_dict
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                safetensors.torch.save_file(
                    state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                )
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen != "no":
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                current_session_num_tokens = self.state.num_input_tokens_seen - self._initial_num_input_tokens_seen
                logs.update(speed_metrics("train", start_time, num_tokens=current_session_num_tokens))

        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def store_flos(self) -> None:
        pass

    def floating_point_ops(self, inputs: dict[str, torch.Tensor | Any]) -> int:
        pass


    def init_hf_repo(self, token: str | None = None) -> None:
        """
        Initializes a git repo in `self.args.hub_model_id`.
        """
        if not self.is_world_process_zero():
            return

        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id

        token = token if token is not None else self.args.hub_token
        repo_url = hf_api().create_repo(repo_name, token=token, private=self.args.hub_private_repo, exist_ok=True)
        self.hub_model_id = repo_url.repo_id
        self.push_in_progress = None

    def create_model_card(
        self,
        language: str | None = None,
        license: str | None = None,
        tags: str | list[str] | None = None,
        model_name: str | None = None,
        finetuned_from: str | None = None,
        tasks: str | list[str] | None = None,
        dataset_tags: str | list[str] | None = None,
        dataset: str | list[str] | None = None,
        dataset_args: str | list[str] | None = None,
    ) -> None:
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `list[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            model_name (`str`, *optional*):
                The name of the model.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `list[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `list[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `list[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `list[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.
        """
        if not self.is_world_process_zero():
            return

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        is_peft_library = False
        if os.path.exists(model_card_filepath):
            library_name = ModelCard.load(model_card_filepath).data.get("library_name")
            is_peft_library = library_name == "peft"

            existing_tags = ModelCard.load(model_card_filepath).data.tags
            if tags is not None and existing_tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                for tag in existing_tags:
                    if tag not in tags:
                        tags.append(tag)

        training_summary = TrainingSummary.from_trainer(
            self,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        model_card = training_summary.to_model_card()
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

        if is_peft_library:
            self.accelerator.unwrap_model(self.model).create_or_update_model_card(self.args.output_dir)

    def push_to_hub(
        self,
        commit_message: str | None = "End of training",
        blocking: bool = True,
        token: str | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> CommitInfo:
        """
        Upload `self.model` and `self.processing_class` to the 🤗 model hub on the repo `self.args.hub_model_id`.

        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                Message to commit while pushing.
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            token (`str`, *optional*, defaults to `None`):
                Token with write permission to overwrite Trainer's original args.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the "main" branch.
            kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to [`~Trainer.create_model_card`].

        Returns:
            The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
            progress of the commit if `blocking=True`.
        """
        self.callback_handler.on_push_begin(self.args, self.state, self.control)

        model_name = kwargs.pop("model_name", None)
        if model_name is None and self.args.should_save:
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                model_name = self.args.hub_model_id.split("/")[-1]
        token = token if token is not None else self.args.hub_token

        if self.hub_model_id is None:
            self.init_hf_repo(token=token)

        self.save_model(_internal_call=True)

        if not self.is_world_process_zero():
            return

        if getattr(self.model, "model_tags", None) is not None:
            if "tags" not in kwargs:
                kwargs["tags"] = []

            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        self.create_model_card(model_name=model_name, **kwargs)

        if revision is None:
            revision = self.args.hub_revision

        self._finish_current_push()

        return hf_api().upload_folder(
            repo_id=self.hub_model_id,
            folder_path=self.args.output_dir,
            commit_message=commit_message,
            token=token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
            revision=revision,
        )

    def _push_from_checkpoint(self, checkpoint_folder: str) -> None:
        pass

    def _finish_current_push(self) -> None:
        """Wait for any in-progress push to the Hub to complete."""
        if not hasattr(self, "push_in_progress"):
            return
        if self.push_in_progress is not None and not self.push_in_progress.is_done():
            logger.info("Waiting for the current checkpoint push to be finished, this might take a couple of minutes.")
            self.push_in_progress.wait_until_done()


    def hyperparameter_search(
        self,
        hp_space: Callable[["optuna.Trial"], dict[str, float]] | None = None,
        compute_objective: Callable[[dict[str, float]], float] | None = None,
        n_trials: int = 20,
        direction: str | list[str] = "minimize",
        backend: str | HPSearchBackend | None = None,
        hp_name: Callable[["optuna.Trial"], str] | None = None,
        **kwargs,
    ) -> BestRun | list[BestRun]:
        pass

    def call_model_init(self, trial: "optuna.Trial | dict[str, Any] | None" = None) -> nn.Module:
        """Invoke `model_init` to get a fresh model instance, optionally conditioned on a hyperparameter trial."""
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    def _hp_search_setup(self, trial: "optuna.Trial | dict[str, Any] | None") -> None:
        """Set up training arguments and accelerator state for a hyperparameter search trial."""
        self._trial = trial

        if self.hp_search_backend is None or trial is None:
            return
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            params = self.hp_space(trial)
        elif self.hp_search_backend == HPSearchBackend.RAY:
            params = trial
            params.pop("wandb", None)
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            params = trial

        for key, value in params.items():
            if not hasattr(self.args, key):
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in"
                    " `TrainingArguments`."
                )
                continue
            old_attr = getattr(self.args, key, None)
            if old_attr is not None:
                value = type(old_attr)(value)

            setattr(self.args, key, value)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info(f"Trial: {trial.params}")
        if self.hp_search_backend == HPSearchBackend.WANDB:
            logger.info(f"W&B Sweep parameters: {trial}")
        if self.is_deepspeed_enabled:
            if self.args.deepspeed is None:
                raise ValueError("For sweeps with deepspeed, `args.deepspeed` must be set")

            self.accelerator.free_memory()

            from accelerate.utils import DeepSpeedPlugin

            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
            self.args.hf_deepspeed_config.trainer_config_process(self.args)
            self.args.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.args.hf_deepspeed_config)

            AcceleratorState()._reset_state()

        self._train_batch_size = self.args.train_batch_size
        self.create_accelerator_and_postprocess()

    def _report_to_hp_search(
        self, trial: "optuna.Trial | dict[str, Any] | None", step: int, metrics: dict[str, float]
    ) -> None:
        pass

    def _tune_save_checkpoint(self, checkpoint_dir: str) -> None:
        """Save a checkpoint during a Ray Tune hyperparameter search trial."""
        output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self.save_model(output_dir, _internal_call=True)
        if self.args.should_save:
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))


    def add_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> None:
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> TrainerCallback | None:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> None:
        pass


    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        if is_sagemaker_mp_enabled():
            return smp.rank() == 0
        return self.args.process_index == 0

    def _move_model_to_device(self, model: nn.Module, device: torch.device) -> None:
        """Move the model to the specified device, re-tying weights on XLA if needed."""
        if getattr(model, "hf_device_map", None) is not None:
            logger.warning(
                "The model is already on multiple devices. Skipping the move to device specified in `args`."
            )
            return
        model = model.to(device)
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()
