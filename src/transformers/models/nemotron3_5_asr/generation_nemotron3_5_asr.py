
from ..nemotron_asr_streaming.generation_nemotron_asr_streaming import (
    NemotronAsrStreamingGenerationMixin,
    NemotronAsrStreamingRNNTDecoderCache,
)


class Nemotron3_5AsrRNNTDecoderCache(NemotronAsrStreamingRNNTDecoderCache): ...


class Nemotron3_5AsrGenerationMixin(NemotronAsrStreamingGenerationMixin):
    def generate(self, inputs=None, generation_config=None, **kwargs):
        self._prompt_ids = kwargs.pop("prompt_ids", None)
        get_audio_features = self.get_audio_features

        def get_audio_features_with_prompt(*args, prompt_ids=None, **features_kwargs):
            pass

        self.get_audio_features = get_audio_features_with_prompt
        try:
            return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        finally:
            del self.get_audio_features
            del self._prompt_ids
