

from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, is_timm_available, requires_backends


if is_timm_available():
    from timm.data import ImageNetInfo, infer_imagenet_subset


@auto_docstring(checkpoint="resnet50")
@strict
class TimmWrapperConfig(PreTrainedConfig):

    model_type = "timm_wrapper"

    architecture: str = "resnet50"
    initializer_range: float = 0.02
    do_pooling: bool = True
    model_args: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        config_dict = config_dict.copy()

        label_names = config_dict.get("label_names")
        is_custom_model = "num_labels" in kwargs or "id2label" in kwargs

        if label_names is None and not is_custom_model:
            requires_backends(cls, ["timm"])
            imagenet_subset = infer_imagenet_subset(config_dict)
            if imagenet_subset:
                dataset_info = ImageNetInfo(imagenet_subset)
                synsets = dataset_info.label_names()
                label_descriptions = dataset_info.label_descriptions(as_dict=True)
                label_names = [label_descriptions[synset] for synset in synsets]

        if label_names is not None and not is_custom_model:
            kwargs["id2label"] = dict(enumerate(label_names))

            if len(set(label_names)) == len(label_names):
                kwargs["label2id"] = {name: i for i, name in enumerate(label_names)}
            else:
                kwargs["label2id"] = None

        num_labels_in_kwargs = kwargs.pop("num_labels", None)
        num_labels_in_dict = config_dict.pop("num_classes", None)

        kwargs["num_labels"] = num_labels_in_kwargs or num_labels_in_dict

        if "pretrained_cfg" in config_dict and "num_classes" in config_dict["pretrained_cfg"]:
            config_dict["pretrained_cfg"].pop("num_classes", None)

        return super().from_dict(config_dict, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output.setdefault("num_classes", self.num_labels)
        output.setdefault("label_names", list(self.id2label.values()))
        output.pop("id2label", None)
        output.pop("label2id", None)
        return output


__all__ = ["TimmWrapperConfig"]
