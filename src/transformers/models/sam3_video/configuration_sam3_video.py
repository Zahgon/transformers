
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3VideoConfig(PreTrainedConfig):

    model_type = "sam3_video"
    is_composition = True
    sub_configs = {
        "detector_config": AutoConfig,
        "tracker_config": AutoConfig,
    }

    detector_config: dict | PreTrainedConfig | None = None
    tracker_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    low_res_mask_size: int = 288
    score_threshold_detection: float = 0.5
    det_nms_thresh: float = 0.1
    assoc_iou_thresh: float = 0.1
    trk_assoc_iou_thresh: float = 0.5
    new_det_thresh: float = 0.7
    recondition_on_trk_masks: bool = True
    hotstart_delay: int = 15
    hotstart_unmatch_thresh: int = 8
    hotstart_dup_thresh: int = 8
    suppress_unmatched_only_within_hotstart: bool = True
    init_trk_keep_alive: int = 30
    max_trk_keep_alive: int = 30
    min_trk_keep_alive: int = -1
    suppress_overlapping_based_on_recent_occlusion_threshold: float = 0.7
    decrease_trk_keep_alive_for_empty_masklets: bool = False
    fill_hole_area: int = 16
    max_num_objects: int = 10000
    recondition_every_nth_frame: int = 16
    high_conf_thresh: float = 0.8
    high_iou_thresh: float = 0.8

    def __post_init__(self, **kwargs):
        if self.detector_config is None:
            self.detector_config = CONFIG_MAPPING["sam3"]()
            logger.info("detector_config is None. Initializing the Sam3Config with default values.")
        if isinstance(self.detector_config, dict):
            self.detector_config["model_type"] = self.detector_config.get("model_type", "sam3")
            self.detector_config = CONFIG_MAPPING[self.detector_config["model_type"]](**self.detector_config)

        if self.tracker_config is None:
            self.tracker_config = CONFIG_MAPPING["sam3_tracker_video"]()
            logger.info("tracker_config is None. Initializing the Sam3TrackerVideoConfig with default values.")
        if isinstance(self.tracker_config, dict):
            self.tracker_config["model_type"] = self.tracker_config.get("model_type", "sam3_tracker_video")
            self.tracker_config = CONFIG_MAPPING[self.tracker_config["model_type"]](**self.tracker_config)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def image_size(self):
        pass

    @image_size.setter
    def image_size(self, value):
        pass


__all__ = ["Sam3VideoConfig"]
