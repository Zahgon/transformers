
from .integrations import (
    is_optuna_available,
    is_ray_tune_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_wandb,
)
from .trainer_utils import (
    HPSearchBackend,
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_wandb,
)
from .utils import logging


logger = logging.get_logger(__name__)


class HyperParamSearchBackendBase:
    name: str
    pip_package: str | None = None

    @staticmethod
    def is_available():
        raise NotImplementedError

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        raise NotImplementedError

    def default_hp_space(self, trial):
        raise NotImplementedError

    def ensure_available(self):
        pass

    @classmethod
    def pip_install(cls):
        pass


class OptunaBackend(HyperParamSearchBackendBase):
    name = "optuna"

    @staticmethod
    def is_available():
        return is_optuna_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_optuna(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        pass


class RayTuneBackend(HyperParamSearchBackendBase):
    name = "ray"
    pip_package = "'ray[tune]'"

    @staticmethod
    def is_available():
        return is_ray_tune_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_ray(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        pass


class WandbBackend(HyperParamSearchBackendBase):
    name = "wandb"

    @staticmethod
    def is_available():
        return is_wandb_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_wandb(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        pass


ALL_HYPERPARAMETER_SEARCH_BACKENDS = {
    HPSearchBackend(backend.name): backend for backend in [OptunaBackend, RayTuneBackend, WandbBackend]
}


def default_hp_search_backend() -> str:
    pass
