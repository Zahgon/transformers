
import collections

from .utils import ExplicitEnum, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class DebugUnderflowOverflow:

    def __init__(self, model, max_frames_to_save=21, trace_batch_nums=None, abort_after_batch_num=None):
        if trace_batch_nums is None:
            trace_batch_nums = []
        self.model = model
        self.trace_batch_nums = trace_batch_nums
        self.abort_after_batch_num = abort_after_batch_num

        self.frames = collections.deque([], max_frames_to_save)
        self.frame = []
        self.batch_number = 0
        self.total_calls = 0
        self.detected_overflow = False
        self.prefix = "                 "

        self.analyse_model()

        self.register_forward_hook()

    def save_frame(self, frame=None):
        pass

    def expand_frame(self, line):
        pass

    def trace_frames(self):
        pass

    def reset_saved_frames(self):
        pass

    def dump_saved_frames(self):
        pass

    def analyse_model(self):
        self.module_names = {m: name for name, m in self.model.named_modules()}

    def analyse_variable(self, var, ctx):
        pass

    def batch_start_frame(self):
        pass

    def batch_end_frame(self):
        pass

    def create_frame(self, module, input, output):
        pass

    def register_forward_hook(self):
        self.model.apply(self._register_forward_hook)

    def _register_forward_hook(self, module):
        pass

    def forward_hook(self, module, input, output):
        pass


def get_abs_min_max(var, ctx):
    pass


def detect_overflow(var, ctx):
    pass


class DebugOption(ExplicitEnum):
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"
