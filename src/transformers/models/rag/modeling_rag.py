
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ...cache_utils import Cache, EncoderDecoderCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationConfig, GenerationMixin, GenerationMode, LogitsProcessorList, StoppingCriteriaList
from ...generation.utils import GENERATION_MODES_MAPPING
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Base class for retriever augmented marginalized models outputs.
    """
)
@dataclass
class RetrievAugLMMarginOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    doc_scores: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    retrieved_doc_embeds: torch.FloatTensor | None = None
    retrieved_doc_ids: torch.LongTensor | None = None
    context_input_ids: torch.LongTensor | None = None
    context_attention_mask: torch.LongTensor | None = None
    question_encoder_last_hidden_state: torch.FloatTensor | None = None
    question_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    question_enc_attentions: tuple[torch.FloatTensor, ...] | None = None
    generator_enc_last_hidden_state: torch.FloatTensor | None = None
    generator_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    generator_enc_attentions: tuple[torch.FloatTensor, ...] | None = None
    generator_dec_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    generator_dec_attentions: tuple[torch.FloatTensor, ...] | None = None
    generator_cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
@dataclass
class RetrievAugLMOutput(ModelOutput):

    logits: torch.FloatTensor | None = None
    doc_scores: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    retrieved_doc_embeds: torch.FloatTensor | None = None
    retrieved_doc_ids: torch.LongTensor | None = None
    context_input_ids: torch.LongTensor | None = None
    context_attention_mask: torch.LongTensor | None = None
    question_encoder_last_hidden_state: torch.FloatTensor | None = None
    question_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    question_enc_attentions: tuple[torch.FloatTensor, ...] | None = None
    generator_enc_last_hidden_state: torch.FloatTensor | None = None
    generator_enc_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    generator_enc_attentions: tuple[torch.FloatTensor, ...] | None = None
    generator_dec_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    generator_dec_attentions: tuple[torch.FloatTensor, ...] | None = None
    generator_cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://huggingface.co/papers/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.
    """
)
@auto_docstring
class RagPreTrainedModel(PreTrainedModel):
    config: RagConfig
    base_model_prefix = "rag"
    _supports_flash_attn = True
    _supports_sdpa = True

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str | None = None,
        generator_pretrained_model_name_or_path: str | None = None,
        retriever: RagRetriever | None = None,
        **kwargs,
    ) -> PreTrainedModel:
        pass


@auto_docstring
class RagModel(RagPreTrainedModel):
    def __init__(
        self,
        config: PreTrainedConfig | None = None,
        question_encoder: PreTrainedModel | None = None,
        generator: PreTrainedModel | None = None,
        retriever: RagRetriever | None = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        r"""
        question_encoder (`PreTrainedModel`, *optional*):
            The model responsible for encoding the question into hidden states for retrieval.
        generator (`PreTrainedModel`, *optional*):
            The model responsible for generating text based on retrieved documents.
        retriever (`RagRetriever`, *optional*):
            The component responsible for retrieving documents from a knowledge base given the encoded question.
        """
        assert config is not None or (question_encoder is not None and generator is not None), (
            "Either a configuration or an question_encoder and a generator has to be provided."
        )

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config)
        if question_encoder is None:
            from ..auto.modeling_auto import AutoModel

            question_encoder = AutoModel.from_config(config.question_encoder)

        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(retriever, RagRetriever), (
                f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            )
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        self.ctx_encoder = None
        self.context_encoder_training = False

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        doc_scores: torch.FloatTensor | None = None,
        context_input_ids: torch.LongTensor | None = None,
        context_attention_mask: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_retrieved: bool | None = None,
        n_docs: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | RetrievAugLMOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [`RagConfig`], used to initialize the model, specifies
            which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
            obtain the indices.

            [What are input IDs?](../glossary#input-ids)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*)
            Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
            *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs *
            sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
            generator's encoder.

            Used by the ([`RagModel`]) model during decoding.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for generation tasks. `None` by default, construct as per instructions for the generator model
            you're using with your RAG instance.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,  target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
            has to be provided to the forward pass. `doc_scores` can be computed via
            `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever. If the model was not initialized with a `retriever` ``context_input_ids` has to be provided to
            the forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`,*optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever. If the model has is not initialized with a `retriever` `context_attention_mask` has to be
            provided to the forward pass. `context_attention_mask` are returned by [`~RagRetriever.__call__`].
        output_retrieved (`bool`, *optional*):
            Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask`. See returned tensors for more detail.
        n_docs (`int`, *optional*):
            The number of documents to retrieve.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, RagModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> outputs = model(input_ids=inputs["input_ids"])
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        if encoder_outputs is None:
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )
                question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.detach().to(device="cpu", dtype=torch.float32).numpy(),
                    prefix=getattr(self.generator.config, "prefix", None),
                    n_docs=n_docs,
                    return_tensors="pt",
                )
                if self.context_encoder_training:
                    (
                        context_input_ids,
                        context_attention_mask,
                        retrieved_doc_embeds,
                        retrieved_doc_input_ids,
                        retrieved_doc_attention_mask,
                        retrieved_doc_ids,
                    ) = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["tokenized_doc_ids"],
                        retriever_outputs["tokenized_doc_attention_mask"],
                        retriever_outputs["doc_ids"],
                    )

                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    retrieved_doc_input_ids = retrieved_doc_input_ids.to(input_ids)
                    retrieved_doc_attention_mask = retrieved_doc_attention_mask.to(input_ids)
                    retrieved_doc_embeds = self.ctx_encoder(
                        retrieved_doc_input_ids, attention_mask=retrieved_doc_attention_mask, return_dict=True
                    ).pooler_output
                    retrieved_doc_embeds = retrieved_doc_embeds.view(
                        -1, n_docs, question_encoder_last_hidden_state.shape[1]
                    )  # reshaping

                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                else:
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["doc_ids"],
                    )

                    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)
            else:
                assert context_input_ids is not None, (
                    "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can"
                    " set a retriever using the `set_retriever(...)` function."
                )
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

        assert doc_scores is not None, (
            "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."
        )

        assert (doc_scores.shape[1] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
        )

        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
            generator_cross_attentions=gen_outputs.cross_attentions,
        )


@auto_docstring(
    custom_intro="""
    A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    """
)
class RagSequenceForGeneration(RagPreTrainedModel):
    def __init__(
        self,
        config: PreTrainedConfig | None = None,
        question_encoder: PreTrainedModel | None = None,
        generator: PreTrainedModel | None = None,
        retriever: RagRetriever | None = None,
        **kwargs,
    ):
        r"""
        question_encoder (`PreTrainedModel`, *optional*):
            The model responsible for encoding the question into hidden states for retrieval.
        generator (`PreTrainedModel`, *optional*):
            The model responsible for generating text based on retrieved documents.
        retriever (`RagRetriever`, *optional*):
            The component responsible for retrieving documents from a knowledge base given the encoded question.
        """
        assert config is not None or (question_encoder is not None and generator is not None), (
            "Either a configuration or an encoder and a generator has to be provided."
        )

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        super().__init__(config)

        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

        self.post_init()

    def set_retriever(self, retriever: RagRetriever):
        pass

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        pass

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        context_input_ids: torch.LongTensor | None = None,
        context_attention_mask: torch.LongTensor | None = None,
        doc_scores: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_retrieved: bool | None = None,
        exclude_bos_score: bool | None = None,
        reduce_loss: bool | None = None,
        labels: torch.LongTensor | None = None,
        n_docs: int | None = None,
        **kwargs,  # needs kwargs for generation
    ) -> RetrievAugLMMarginOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [`RagConfig`], used to initialize the model, specifies
            which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
            obtain the indices.

            [What are input IDs?](../glossary#input-ids)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*)
            Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
            *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs *
            sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
            generator's encoder.

            Used by the ([`RagModel`]) model during decoding.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for generation tasks. `None` by default, construct as per instructions for the generator model
            you're using with your RAG instance.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,  target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever. If the model was not initialized with a `retriever` ``context_input_ids` has to be provided to
            the forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`,*optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever. If the model has is not initialized with a `retriever` `context_attention_mask` has to be
            provided to the forward pass. `context_attention_mask` are returned by [`~RagRetriever.__call__`].
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
            has to be provided to the forward pass. `doc_scores` can be computed via
            `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
        output_retrieved (`bool`, *optional*):
            Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask`. See returned tensors for more detail.
        exclude_bos_score (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the score of the BOS token is disregarded when computing
            the loss.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
            operation.
        n_docs (`int`, *optional*):
            The number of documents to retrieve.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]
        >>> labels = targets["input_ids"]
        >>> outputs = model(input_ids=input_ids, labels=labels)

        >>> # or use retriever separately
        >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
        >>> # 1. Encode
        >>> question_hidden_states = model.question_encoder(input_ids)[0]
        >>> # 2. Retrieve
        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        >>> doc_scores = torch.bmm(
        ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ... ).squeeze(1)
        >>> # 3. Forward to generator
        >>> outputs = model(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ...     decoder_input_ids=labels,
        ... )
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        if labels is not None:
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                decoder_input_ids,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                exclude_bos_score=exclude_bos_score,
                n_docs=n_docs,
            )

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=outputs.logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        context_input_ids: torch.LongTensor | None = None,
        context_attention_mask: torch.LongTensor | None = None,
        doc_scores: torch.FloatTensor | None = None,
        do_deduplication: bool | None = None,  # defaults to True
        num_return_sequences: int | None = None,  # defaults to 1
        num_beams: int | None = None,  # defaults to 1
        n_docs: int | None = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Implements RAG sequence "thorough" decoding. Read the [`~generation.GenerationMixin.generate`]` documentation
        for more information on how to set other generate input parameters.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model is not initialized with a `retriever` or `input_ids` is not given, `context_input_ids` and
                `context_attention_mask` have to be provided to the forward pass. They are returned by
                [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model is not initialized with a `retriever` or `input_ids` is not given, `doc_scores` has to be
                provided to the forward pass. `doc_scores` are returned by [`~RagRetriever.__call__`].
            do_deduplication (`bool`, *optional*):
                Whether or not to deduplicate the generations from different context documents for a given input. Has
                to be set to `False` if used while training with distributed backend.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation.GenerationMixin.generate`]` function,
                where we set `num_return_sequences` to `num_beams`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            kwargs (`dict[str, Any]`, *optional*):
                Additional kwargs will be passed to [`~generation.GenerationMixin.generate`].

        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else getattr(self.config, "num_return_sequences", 1)
        )
        num_beams = num_beams if num_beams is not None else getattr(self.config, "num_beams", 1)

        assert input_ids is not None or context_input_ids is not None, (
            " At least one of input_ids or context_input_ids must be given"
        )

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(
                input_ids,
                question_hidden_states.detach().to(device="cpu", dtype=torch.float32).numpy(),
                prefix=getattr(self.generator.config, "prefix", None),
                n_docs=n_docs,
                return_tensors="pt",
            )["context_input_ids"]

            context_input_ids = context_input_ids.to(input_ids)

        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams
        model_kwargs["attention_mask"] = None

        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs

        for index in range(batch_size):
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)

            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            if do_deduplication:
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))

            num_candidates = output_sequences.shape[
                0
            ]  # after deduplication, this number can be less than n_docs*n_beam

            if input_ids is not None:
                new_input_ids = input_ids[index : index + 1].repeat(num_candidates, 1)
                outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            else:  # input_ids is None, need context_input_ids/mask and doc_scores
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

                individual_input_ids = generator_input_ids.repeat(
                    num_candidates, 1
                )  # (num_candidates*n_docs, max_len)

                individual_attention_mask = context_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)

                individual_doc_scores = doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]

                outputs = self(
                    context_input_ids=individual_input_ids,
                    context_attention_mask=individual_attention_mask,
                    doc_scores=individual_doc_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                )

            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        output = tensors[0].new(sum(t.shape[0] for t in tensors), max(t.shape[1] for t in tensors)).fill_(pad_token_id)
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]] = t
            ind += t.shape[0]
        return output


@auto_docstring(
    custom_intro="""
    A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """
)
class RagTokenForGeneration(RagPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: PreTrainedConfig | None = None,
        question_encoder: PreTrainedModel | None = None,
        generator: PreTrainedModel | None = None,
        retriever: RagRetriever | None = None,
        **kwargs,
    ):
        r"""
        question_encoder (`PreTrainedModel`, *optional*):
            The model responsible for encoding the question into hidden states for retrieval.
        generator (`PreTrainedModel`, *optional*):
            The model responsible for generating text based on retrieved documents.
        retriever (`RagRetriever`, *optional*):
            The component responsible for retrieving documents from a knowledge base given the encoded question.
        """
        assert config is not None or (question_encoder is not None and generator is not None), (
            "Either a configuration or an encoder and a generator has to be provided."
        )

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

        self.post_init()

    def set_retriever(self, retriever: RagRetriever):
        pass

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        pass

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        doc_scores=None,
        n_docs=None,
        **kwargs,
    ):

        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "doc_scores": doc_scores,
            "context_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "do_marginalize": True,
            "n_docs": n_docs,
        }

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        pass

    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        context_input_ids: torch.LongTensor | None = None,
        context_attention_mask: torch.LongTensor | None = None,
        doc_scores: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_retrieved: bool | None = None,
        do_marginalize: bool | None = None,
        reduce_loss: bool | None = None,
        labels: torch.LongTensor | None = None,
        n_docs: int | None = None,
        **kwargs,  # needs kwargs for generation
    ) -> RetrievAugLMMarginOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [`RagConfig`], used to initialize the model, specifies
            which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
            obtain the indices.

            [What are input IDs?](../glossary#input-ids)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*)
            Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
            *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs *
            sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
            generator's encoder.

            Used by the ([`RagModel`]) model during decoding.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for generation tasks. `None` by default, construct as per instructions for the generator model
            you're using with your RAG instance.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,  target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever. If the model was not initialized with a `retriever` ``context_input_ids` has to be provided to
            the forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`,*optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever. If the model has is not initialized with a `retriever` `context_attention_mask` has to be
            provided to the forward pass. `context_attention_mask` are returned by [`~RagRetriever.__call__`].
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
            has to be provided to the forward pass. `doc_scores` can be computed via
            `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
        output_retrieved (`bool`, *optional*):
            Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask`. See returned tensors for more detail.
        do_marginalize (`bool`, *optional*):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
            operation.
        n_docs (`int`, *optional*):
            The number of documents to retrieve.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]
        >>> labels = targets["input_ids"]
        >>> outputs = model(input_ids=input_ids, labels=labels)

        >>> # or use retriever separately
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
        >>> # 1. Encode
        >>> question_hidden_states = model.question_encoder(input_ids)[0]
        >>> # 2. Retrieve
        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        >>> doc_scores = torch.bmm(
        ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ... ).squeeze(1)
        >>> # 3. Forward to generator
        >>> outputs = model(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ...     decoder_input_ids=labels,
        ... )

        >>> # or directly generate
        >>> generated = model.generate(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ... )
        >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        context_input_ids: torch.LongTensor | None = None,
        context_attention_mask: torch.LongTensor | None = None,
        doc_scores: torch.FloatTensor | None = None,
        n_docs: int | None = None,
        generation_config: GenerationConfig | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        logits_processor: LogitsProcessorList | None = LogitsProcessorList(),
        stopping_criteria: StoppingCriteriaList | None = StoppingCriteriaList(),
        **kwargs,
    ) -> torch.LongTensor:
        """
        Implements RAG token decoding.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID
                `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on
                the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for
                constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://huggingface.co/papers/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and a
                model's config. If a logit processor is passed that is already created with the arguments or a model's
                config an error is thrown.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                model's config. If a stopping criteria is passed that is already created with the arguments or a
                model's config an error is thrown.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """
        generation_mode_kwargs = self._extract_generation_mode_kwargs(None, kwargs, False, None, None)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        generation_mode = generation_config.get_generation_mode()
        if generation_mode not in [
            GenerationMode.SAMPLE,
            GenerationMode.GREEDY_SEARCH,
            GenerationMode.BEAM_SEARCH,
            GenerationMode.BEAM_SAMPLE,
        ]:
            raise ValueError(
                f"RAG model is not compatible with {generation_mode} generation. Please check your generation parameters."
            )
        decoding_method = getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask)

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(
                input_ids,
                question_hidden_states.detach().to(device="cpu", dtype=torch.float32).numpy(),
                prefix=getattr(self.generator.config, "prefix", None),
                n_docs=n_docs,
                return_tensors="pt",
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)

            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(
                1
            )

        assert (context_input_ids.shape[0] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
        )

        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)

        input_ids = torch.full(
            (batch_size * generation_config.num_beams, 1),
            generation_config.decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        input_ids_seq_length = input_ids.shape[-1]
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=generation_config.num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(
            last_hidden_state, num_beams=generation_config.num_beams
        )

        doc_scores = doc_scores.repeat_interleave(generation_config.num_beams, dim=0)

        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs
        model_kwargs["use_cache"] = generation_config.use_cache

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=context_input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=input_ids.device,
        )

        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        self._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            generation_mode=None,
            batch_size=input_ids.shape[0],
            max_cache_length=generation_config.max_length - 1,
        )

        return decoding_method(
            self,
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            **generation_mode_kwargs,
            **model_kwargs,
        )

    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        pass

    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.rag.generator.set_output_embeddings(new_embeddings)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


__all__ = ["RagModel", "RagPreTrainedModel", "RagSequenceForGeneration", "RagTokenForGeneration"]
