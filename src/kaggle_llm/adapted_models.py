from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers import AutoModelForMultipleChoice, AutoConfig
from pathlib import Path
from typing import *
import torch.nn as nn
import torch

from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout, DebertaV2ForMultipleChoice, DebertaV2Model, DebertaV2PreTrainedModel


NUM_CHOICES = 5


class LlamaModelForMultipleChoice(LlamaPreTrainedModel):
    _keep_in_fp32_modules = [
        # "pooler",
        # "dropout",
        "classifier",
        # "classifier0",
        # "classifier1",
    ]

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel(config)

        config.pooler_hidden_size = getattr(config, "pooler_hidden_size", config.hidden_size)
        config.pooler_dropout = 0
        config.pooler_hidden_act = "gelu"
        # self.pooler = ContextPooler(config)
        # output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(4096, 1)
        # self.embed_dims = 4096
        # self.classifier0 = nn.Linear(self.embed_dims, 128, bias=False)
        # self.classifier1 = nn.Linear(128*NUM_CHOICES, NUM_CHOICES)
        drop_out = getattr(config, "cls_dropout", 0)
        self.dropout = StableDropout(drop_out)
        self.init_weights()

    def save_extra_modules(self, checkpoint_dir: Union[str, Path]):
        checkpoint_dir = Path(checkpoint_dir)
        # torch.save(self.classifier.state_dict(), checkpoint_dir / "classifier.pt")
        torch.save(self.classifier0.state_dict(), checkpoint_dir / "classifier0.pt")
        torch.save(self.classifier1.state_dict(), checkpoint_dir / "classifier1.pt")

    def load_extra_modules(self, checkpoint_dir: Union[str, Path]):
        checkpoint_dir = Path(checkpoint_dir)
        # self.classifier.load_state_dict(torch.load(checkpoint_dir / "classifier.pt"))
        self.classifier0.load_state_dict(torch.load(checkpoint_dir / "classifier0.pt"))
        self.classifier1.load_state_dict(torch.load(checkpoint_dir / "classifier1.pt"))

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        stop_token_indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        batch_size = input_ids.shape[0]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        #
        outputs = self.model.forward(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            position_ids=flat_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # outputs = self.model.forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     inputs_embeds=inputs_embeds,
        #     position_ids=position_ids,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        encoder_layer = outputs[0]
        # pooled_output = self.pooler(encoder_layer)
        # pooled_output = encoder_layer[:, 0]

        if input_ids is not None:
            sequence_lengths = (torch.ne(flat_input_ids, self.config.pad_token_id).sum(-1) - 1).to(encoder_layer.device)
        else:
            sequence_lengths = -1
        pooled_output = encoder_layer[
            torch.arange(encoder_layer.shape[0], device=encoder_layer.device),
            sequence_lengths,
            :
        ]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        # pooled_output = torch.gather(encoder_layer, 1, stop_token_indices.unsqueeze(-1).tile(1, 1, self.embed_dims))
        # compressed_output = self.classifier0(pooled_output).reshape((batch_size, -1))
        # compressed_output = torch.nn.functional.gelu(compressed_output)
        # reshaped_logits = self.classifier1(compressed_output)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(reshaped_logits, labels)
        # if labels is not None:
        #     one_hot_label = torch.nn.functional.one_hot(labels, num_classes=5)
        #     loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #         logits.reshape(-1, 1),
        #         one_hot_label.float().reshape(-1, 1)
        #     )

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        assert not torch.isnan(reshaped_logits).any(), f"found nan"
        if loss is not None:
            assert not torch.isnan(loss).any(), f"loss nan"
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoModelForMultipleChoice.register(LlamaConfig, LlamaModelForMultipleChoice)


# =====


class DebertaV2ForMultipleChoice2(DebertaV2PreTrainedModel):
    # _keep_in_fp32_modules = [
    #     # "pooler",
    #     # "dropout",
    #     "classifier",
    # ]

    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)

        num_choices = 5
        self.classifier0 = nn.Linear(1024, 256, bias=False)
        self.classifier1 = nn.Linear(256*num_choices, num_choices)
        # self.classifier = nn.Linear(1024, 5)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        stop_token_indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        batch_size = input_ids.shape[0]

        # flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # flat_inputs_embeds = (
        #     inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
        #     if inputs_embeds is not None
        #     else None
        # )
        #
        # outputs = self.deberta(
        #     flat_input_ids,
        #     position_ids=flat_position_ids,
        #     token_type_ids=flat_token_type_ids,
        #     attention_mask=flat_attention_mask,
        #     inputs_embeds=flat_inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        outputs = self.deberta(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]

        # if input_ids is not None:
        #     # sequence_lengths = (torch.ne(flat_input_ids, self.config.pad_token_id).sum(-1) - 1).to(encoder_layer.device)
        #     sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(encoder_layer.device)
        # else:
        #     sequence_lengths = -1
        # pooled_output = encoder_layer[
        #     torch.arange(encoder_layer.shape[0], device=encoder_layer.device),
        #     sequence_lengths,
        #     :
        # ]
        # logits = self.classifier(pooled_output)
        # reshaped_logits = logits.view(-1, num_choices)

        # take 5 tokens from the [SEP] locations
        pooled_output = torch.gather(encoder_layer, 1, stop_token_indices.unsqueeze(-1).tile(1, 1, 1024))
        compressed_output = self.classifier0(pooled_output).reshape((batch_size, -1))
        compressed_output = torch.nn.functional.gelu(compressed_output)
        reshaped_logits = self.classifier1(compressed_output)

        # reshaped_logits = self.classifier(pooled_output)

        # compressed_output = self.classifier0(pooled_output.reshape((batch_size, num_choices, -1))).reshape((batch_size, -1))
        # compressed_output = torch.nn.functional.gelu(compressed_output)
        # reshaped_logits = self.classifier1(compressed_output)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                reshaped_logits,
                labels
            )
        # if labels is not None:
        #     one_hot_label = torch.nn.functional.one_hot(labels, num_classes=5)
        #     loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #         logits.reshape(-1, 1),
        #         one_hot_label.float().reshape(-1, 1)
        #     )

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoModelForMultipleChoice.register(DebertaV2Config, DebertaV2ForMultipleChoice2, exist_ok=True)
