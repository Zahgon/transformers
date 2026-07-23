
import dataclasses
import json
import math
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from .trainer_utils import IntervalStrategy, SaveStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging


logger = logging.get_logger(__name__)


@dataclass
class TrainerState:

    epoch: float = 0
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    train_batch_size: int | None = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    log_history: list[dict[str, float]] = None
    best_metric: float | None = None
    best_global_step: int | None = None
    best_model_checkpoint: str | None = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str | None = None
    trial_params: dict[str, str | float | int | bool] | None = None
    stateful_callbacks: list["TrainerCallback"] | None = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.stateful_callbacks is None:
            self.stateful_callbacks = {}
        elif isinstance(self.stateful_callbacks, dict):
            pass
        else:
            stateful_callbacks = {}
            for callback in self.stateful_callbacks:
                if not isinstance(callback, (ExportableState)):
                    raise TypeError(
                        f"All callbacks passed to be saved must inherit `ExportableState`, but received {type(callback)}"
                    )
                name = callback.__class__.__name__
                if name in stateful_callbacks:
                    if not isinstance(stateful_callbacks[name], list):
                        stateful_callbacks[name] = [stateful_callbacks[name]]
                    stateful_callbacks[name].append(callback.state())
                else:
                    stateful_callbacks[name] = callback.state()
            self.stateful_callbacks = stateful_callbacks

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    def compute_steps(self, args, max_steps):
        pass

    def init_training_references(self, trainer, max_steps, num_train_epochs, trial):
        pass


class ExportableState:

    def state(self) -> dict:
        raise NotImplementedError("You must implement a `state` function to utilize this class.")

    @classmethod
    def from_state(cls, state):
        pass


@dataclass
class TrainerControl(ExportableState):

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        pass

    def _new_epoch(self):
        pass

    def _new_step(self):
        pass

    def state(self) -> dict:
        return {
            "args": {
                "should_training_stop": self.should_training_stop,
                "should_epoch_stop": self.should_epoch_stop,
                "should_save": self.should_save,
                "should_evaluate": self.should_evaluate,
                "should_log": self.should_log,
            },
            "attributes": {},
        }


class TrainerCallback:

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called before the optimizer step but after gradient clipping. Useful for monitoring gradients.
        """

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.
        """

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """

    def on_push_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called before pushing the model to the hub, at the beginning of Trainer.push_to_hub and Trainer._push_from_checkpoint.
        """


class CallbackHandler(TrainerCallback):

    def __init__(self, callbacks, model, processing_class, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        pass

    @property
    def callback_list(self):
        pass

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_init_end", args, state, control, **kwargs)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics, **kwargs)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        return self.call_event("on_predict", args, state, control, metrics=metrics, **kwargs)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs, **kwargs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_prediction_step", args, state, control, **kwargs)

    def on_push_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_push_begin", args, state, control, **kwargs)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass


class ProgressCallback(TrainerCallback):

    def __init__(self, max_str_len: int = 100):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.training_bar = None
        self.prediction_bar = None
        self.max_str_len = max_str_len

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                elif isinstance(v, float):
                    shallow_logs[k] = f"{v:.4g}"
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            self.training_bar.write(str(shallow_logs))

    def on_train_end(self, args, state, control, **kwargs):
        pass


class PrinterCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if logs is not None:
                logs = {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in logs.items()}
            print(logs)


class EarlyStoppingCallback(TrainerCallback, ExportableState):

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float | None = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

    def state(self) -> dict:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
            },
        }
