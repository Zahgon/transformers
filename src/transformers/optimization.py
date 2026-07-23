
from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from .trainer_pt_utils import LayerWiseDummyOptimizer, LayerWiseDummyScheduler
from .trainer_utils import SchedulerType
from .utils import logging


logger = logging.get_logger(__name__)


def _get_constant_lambda(_=None):
    pass


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    pass


def get_reduce_on_plateau_schedule(optimizer: Optimizer, **kwargs):
    pass


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    pass


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    pass


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    pass


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    pass


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    pass


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    pass


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    pass


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    pass


def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    pass


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    pass


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int | None = None):
    pass


def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: int | None = None, last_epoch: int = -1
):
    pass


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    pass


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float | None = None,
    min_lr_rate: float | None = None,
):
    pass


def _get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float = 0.0,
    warmup_lr_rate: float | None = None,
):
    pass


def get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float | None = None,
    min_lr_rate: float | None = None,
    warmup_lr_rate: float | None = None,
):
    pass


def _get_wsd_scheduler_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_type: str,
    decay_type: str,
    min_lr_ratio: float,
    num_cycles: float,
):
    pass


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: int | None = None,
    num_stable_steps: int | None = None,
    warmup_type: str = "linear",
    decay_type: str = "cosine",
    min_lr_ratio: float = 0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    pass


class StreamingAverage:

    def __init__(self, window_size: int) -> None:
        self.window_size: int = window_size
        self.values: list[float] = []
        self.sum: float = 0.0

    def streamavg(self, value: float) -> float:
        """Add a value and return the current rolling average."""
        self.values.append(value)
        self.sum += value

        if len(self.values) > self.window_size:
            removed = self.values.pop(0)
            self.sum -= removed

        return self.sum / len(self.values)

    def state_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "values": self.values.copy(),
            "sum": self.sum,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.window_size = state_dict.get("window_size", self.window_size)
        self.values = state_dict.get("values", []).copy()
        self.sum = state_dict.get("sum", 0.0)


class GreedyLR:

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.95,
        patience: int = 10,
        threshold: float = 1e-6,
        threshold_mode: str = "abs",
        cooldown: int = 0,
        warmup: int = 0,
        min_lr: float | list[float] = 1e-3,
        max_lr: float | list[float] = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        smooth: bool = False,
        window_size: int = 50,
        reset_start: int = 500,
    ) -> None:
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.warmup = warmup
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.eps = eps
        self.smooth = smooth
        self.window_size = window_size
        self.reset_start = reset_start
        self.reset_start_original = reset_start
        self.last_epoch = 0

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self._init_lrs = [group["lr"] for group in optimizer.param_groups]
        self._last_lr = self._init_lrs.copy()

        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.num_good_epochs = 0

        if mode not in ("min", "max"):
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")

        self._streaming_avg: StreamingAverage | None = None
        if smooth:
            self._streaming_avg = StreamingAverage(window_size)

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Perform a scheduler step based on the given metrics.

        Args:
            metrics (`float`):
                The metric value to use for LR adjustment decisions.
            epoch (`int`, *optional*):
                The current epoch number. If None, uses internal counter.
        """
        current = float(metrics)

        if self.smooth and self._streaming_avg is not None:
            current = self._streaming_avg.streamavg(current)

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            self.num_good_epochs = 0
        elif self.warmup_counter > 0:
            self.warmup_counter -= 1
            self.num_bad_epochs = 0
            self.num_good_epochs = 0
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
                self.num_good_epochs += 1
            else:
                self.num_bad_epochs += 1
                self.num_good_epochs = 0

            if self.num_good_epochs > self.patience:
                self._increase_lr(epoch)
                self.warmup_counter = self.warmup
                self.num_good_epochs = 0
            elif self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return current < best * (1.0 - self.threshold)
            else:
                return current < best - self.threshold
        else:
            if self.threshold_mode == "rel":
                return current > best * (1.0 + self.threshold)
            else:
                return current > best + self.threshold

    def _reduce_lr(self, epoch: int) -> None:
        all_at_min = True
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])

            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.")

            if param_group["lr"] > self.min_lrs[i]:
                all_at_min = False

        if all_at_min:
            self.reset_start -= 1
            if self.reset_start <= 0:
                self._reset()

    def _increase_lr(self, epoch: int) -> None:
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = min(old_lr / self.factor, self.max_lrs[i])

            if new_lr - old_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Epoch {epoch}: increasing learning rate of group {i} to {new_lr:.4e}.")

        self.reset_start = self.reset_start_original

    def _reset(self) -> None:
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self._init_lrs[i]

        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self.reset_start = self.reset_start_original

        if self.smooth and self._streaming_avg is not None:
            self._streaming_avg = StreamingAverage(self.window_size)

        if self.verbose:
            print("Scheduler reset to initial state.")

    def get_last_lr(self) -> list[float]:
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a dictionary."""
        state = {
            "factor": self.factor,
            "min_lrs": self.min_lrs,
            "max_lrs": self.max_lrs,
            "patience": self.patience,
            "verbose": self.verbose,
            "cooldown": self.cooldown,
            "warmup": self.warmup,
            "cooldown_counter": self.cooldown_counter,
            "warmup_counter": self.warmup_counter,
            "mode": self.mode,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "num_good_epochs": self.num_good_epochs,
            "eps": self.eps,
            "last_epoch": self.last_epoch,
            "smooth": self.smooth,
            "window_size": self.window_size,
            "reset_start": self.reset_start,
            "reset_start_original": self.reset_start_original,
            "_last_lr": self._last_lr,
            "_init_lrs": self._init_lrs,
        }

        if self.smooth and self._streaming_avg is not None:
            state["_streaming_avg"] = self._streaming_avg.state_dict()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a dictionary."""
        self.factor = state_dict.get("factor", self.factor)
        self.min_lrs = state_dict.get("min_lrs", self.min_lrs)
        self.max_lrs = state_dict.get("max_lrs", self.max_lrs)
        self.patience = state_dict.get("patience", self.patience)
        self.verbose = state_dict.get("verbose", self.verbose)
        self.cooldown = state_dict.get("cooldown", self.cooldown)
        self.warmup = state_dict.get("warmup", self.warmup)
        self.cooldown_counter = state_dict.get("cooldown_counter", self.cooldown_counter)
        self.warmup_counter = state_dict.get("warmup_counter", self.warmup_counter)
        self.mode = state_dict.get("mode", self.mode)
        self.threshold = state_dict.get("threshold", self.threshold)
        self.threshold_mode = state_dict.get("threshold_mode", self.threshold_mode)
        self.best = state_dict.get("best", self.best)
        self.num_bad_epochs = state_dict.get("num_bad_epochs", self.num_bad_epochs)
        self.num_good_epochs = state_dict.get("num_good_epochs", self.num_good_epochs)
        self.eps = state_dict.get("eps", self.eps)
        self.last_epoch = state_dict.get("last_epoch", self.last_epoch)
        self.smooth = state_dict.get("smooth", self.smooth)
        self.window_size = state_dict.get("window_size", self.window_size)
        self.reset_start = state_dict.get("reset_start", self.reset_start)
        self.reset_start_original = state_dict.get("reset_start_original", self.reset_start_original)
        self._last_lr = state_dict.get("_last_lr", self._last_lr)
        self._init_lrs = state_dict.get("_init_lrs", self._init_lrs)

        if "_streaming_avg" in state_dict:
            if self._streaming_avg is None:
                self._streaming_avg = StreamingAverage(self.window_size)
            self._streaming_avg.load_state_dict(state_dict["_streaming_avg"])

        if "_last_lr" in state_dict:
            for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group["lr"] = lr


def get_greedy_schedule(optimizer: Optimizer, **kwargs):
    pass


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
    SchedulerType.COSINE_WITH_MIN_LR: get_cosine_with_min_lr_schedule_with_warmup,
    SchedulerType.COSINE_WARMUP_WITH_MIN_LR: get_cosine_with_min_lr_schedule_with_warmup_lr_rate,
    SchedulerType.WARMUP_STABLE_DECAY: get_wsd_schedule,
    SchedulerType.GREEDY: get_greedy_schedule,
}


def get_scheduler(
    name: str | SchedulerType,
    optimizer: Optimizer,
    num_warmup_steps: int | None = None,
    num_training_steps: int | None = None,
    scheduler_specific_kwargs: dict | None = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        scheduler_specific_kwargs (`dict`, *optional*):
            Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
            parameters will cause the scheduler function to raise a TypeError.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if optimizer is not None and isinstance(optimizer, LayerWiseDummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict = {}

        for param in optimizer_dict:
            scheduler_dict[param] = get_scheduler(
                name,
                optimizer=optimizer_dict[param],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

        def scheduler_hook(param):
            pass

        for param in optimizer_dict:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(scheduler_hook)

        return LayerWiseDummyScheduler(optimizer_dict=optimizer_dict, lr=optimizer.defaults["lr"])

    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    if name == SchedulerType.GREEDY:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **scheduler_specific_kwargs)

    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **scheduler_specific_kwargs,
        )

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )


class Adafactor(Optimizer):

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss


class AdafactorSchedule(LambdaLR):

    def __init__(self, optimizer, initial_lr=0.0):
        def lr_lambda(_):
            pass

        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        pass


def get_adafactor_schedule(optimizer, initial_lr=0.0):
    pass
