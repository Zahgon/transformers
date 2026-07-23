
import torch

from ..trainer_utils import _is_peft_model


def neftune_post_forward_hook(module, input, output):
    pass


def activate_neftune(model, neftune_noise_alpha, accelerator=None):
    """
    Activates NEFTune (Noisy Embeddings for Fine-Tuning) on the model.

    NEFTune adds noise to embedding vectors during training, which has been shown to improve
    fine-tuning performance. See https://huggingface.co/papers/2310.05914 for details.

    Args:
        model (`torch.nn.Module`):
            The model to activate NEFTune on.
        neftune_noise_alpha (`float`):
            The noise alpha value controlling the magnitude of the noise.
        accelerator (`Accelerator`, *optional*):
            The accelerator instance. If provided, the model will be unwrapped before
            accessing embeddings. Required when using distributed training.

    Returns:
        `torch.utils.hooks.RemovableHandle`: The hook handle that can be used to deactivate NEFTune.
    """
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        unwrapped_model = model

    if _is_peft_model(unwrapped_model):
        embeddings = unwrapped_model.base_model.model.get_input_embeddings()
    else:
        embeddings = unwrapped_model.get_input_embeddings()

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)

    return hook_handle


def deactivate_neftune(model, hook_handle, accelerator=None):
    pass
