import os
import signal
import threading

from .trainer_callback import TrainerCallback
from .trainer_utils import PREFIX_CHECKPOINT_DIR
from .utils import logging


logger = logging.get_logger(__name__)


class CheckpointManager:
    def __init__(self, trainer, kill_wait: int = 3):
        """
        Initialize the CheckpointManager for Just-In-Time checkpoint handling.

        Args:
            trainer: The Trainer instance that will be used to save checkpoints when SIGTERM is received.
            kill_wait (`int`, *optional*, defaults to 3): Grace period to distinguish between SIGTERM and SIGKILL.
        """
        self.trainer = trainer
        self.is_checkpoint_requested = False
        self._original_sigterm_handler = None
        self.kill_wait = kill_wait

    def setup_signal_handler(self):
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        logger.info("JIT checkpoint signal handler registered for SIGTERM")

    def _sigterm_handler(self, signum, frame):
        pass

    def _enable_checkpoint(self):
        pass

    def execute_jit_checkpoint(self):
        pass


class JITCheckpointCallback(TrainerCallback):

    def __init__(self):
        self.trainer = None
        self.jit_manager: CheckpointManager | None = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        if trainer.args.enable_jit_checkpoint:
            self.jit_manager = CheckpointManager(trainer=trainer)
            self.jit_manager.setup_signal_handler()
            logger.info("JIT checkpointing enabled")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass
