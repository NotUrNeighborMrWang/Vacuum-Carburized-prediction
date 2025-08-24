
import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from transformers.models.flava import *
import torch
import torch.utils.checkpoint
from torch import nn

PossibleConfigs = Union[FlavaTextConfig, FlavaImageConfig, FlavaMultimodalConfig]


class ModelsOutput(nn.Module):
    def __init__(self, config: PossibleConfigs) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # Copied from transformers.models.vit.modeling_vit.ViTOutput.forward
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states
    

class Model():
    config_class = FlavaConfig

    def __init__(self, config: FlavaConfig):
        super().__init__(config)

        if not isinstance(config.text_config, FlavaTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type FlavaTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.image_config, FlavaImageConfig):
            raise TypeError(
                "config.image_config is expected to be of type FlavaImageConfig but is of type"
                f" {type(config.image_config)}."
            )

        if not isinstance(config.multimodal_config, FlavaMultimodalConfig):
            raise TypeError(
                "config.multimodal_config is expected to be of type FlavaMultimodalConfig but "
                + f"is of type {type(config.multimodal_config)}."
            )

        text_config = config.text_config
        image_config = config.image_config
        multimodal_config = config.multimodal_config

        self.projection_dim = config.projection_dim
        self.text_hidden_size = text_config.hidden_size
        self.image_hidden_size = image_config.hidden_size
        self.mm_hidden_size = multimodal_config.hidden_size

        self.text_model = FlavaTextModel(text_config)
        self.image_model = FlavaImageModel(image_config)
        self.multimodal_model = FlavaMultimodalModel(multimodal_config)

        self.image_projection = nn.Linear(self.image_hidden_size, self.projection_dim)
        self.text_projection = nn.Linear(self.text_hidden_size, self.projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.image_to_mm_projection = nn.Linear(self.image_hidden_size, self.mm_hidden_size)
        self.text_to_mm_projection = nn.Linear(self.text_hidden_size, self.mm_hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAVA_TEXT_INPUTS_DOCSTRING.format("batch_size, text_seq_length"))
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`FlavaTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoProcessor, FlavaModel

        >>> model = FlavaModel.from_pretrained("{0}")
        >>> processor = AutoProcessor.from_pretrained("{0}")

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], max_length=77, padding="max_length", return_tensors="pt"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```""".format(_CHECKPOINT_FOR_DOC)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[0]  # last_hidden_state
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(FLAVA_IMAGE_INPUTS_DOCSTRING.format("batch_size, image_num_patches"))
    def get_image_features(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`FlavaImageModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FlavaModel

        >>> model = FlavaModel.from_pretrained("{0}")
        >>> processor = AutoProcessor.from_pretrained("{0}")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```""".format(_CHECKPOINT_FOR_DOC)
        image_outputs = self.image_model(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = image_outputs[0]  # last_hidden_state
        image_features = self.image_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        skip_multimodal_encoder: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: bool = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelsOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FlavaModel

        >>> model = FlavaModel.from_pretrained("facebook/flava-full")
        >>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)

        >>> image_embeddings = outputs.image_embeddings
        >>> text_embeddings = outputs.text_embeddings
        >>> multimodal_embeddings = outputs.multimodal_embeddings

        >>> outputs.image_embeddings.shape
        torch.Size([1, 197, 768])

        >>> text_embeddings.shape
        torch.Size([1, 7, 768])

        >>> multimodal_embeddings.shape
        torch.Size([1, 205, 768])
        ```
        """

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if not output_hidden_states:
            raise ValueError("FLAVA model requires hidden states to work. Please set `output_hidden_states=True`")
        image_embeddings = None
        image_states = None
        image_mm_projection = None
        image_output = None
        if pixel_values is not None:
            image_output = self.image_model(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
                attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings, image_states = image_output[0], image_output[2]
            # Note that these states don't use final layernorm in the transformer model
            image_mm_projection = self.image_to_mm_projection(image_states[-1])

        text_embeddings = None
        text_states = None
        text_mm_projection = None
        text_output = None
        if input_ids is not None:
            text_output = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            text_embeddings, text_states = text_output[0], text_output[2]
            # Note that these states don't use final layernorm in the transformer model
            text_mm_projection = self.text_to_mm_projection(text_states[-1])

        multimodal_embeddings = None
        multimodal_output = None
        if image_mm_projection is not None and text_mm_projection is not None and not skip_multimodal_encoder:
            if attention_mask is not None:
                batch_size, seq_len, _ = image_mm_projection.shape
                if self.multimodal_model.use_cls_token:
                    seq_len += 1
                attention_mask_image = torch.ones(batch_size, seq_len, device=image_mm_projection.device)
                attention_multimodal = torch.cat([attention_mask_image, attention_mask], dim=1)
            else:
                attention_multimodal = None
            multimodal_input = torch.cat([image_mm_projection, text_mm_projection], dim=1)
            multimodal_output = self.multimodal_model(
                multimodal_input, attention_mask=attention_multimodal, return_dict=return_dict
            )
            multimodal_embeddings = multimodal_output[0]

        if not return_dict:
            return (
                image_embeddings,
                image_output,
                text_embeddings,
                text_output,
                multimodal_embeddings,
                multimodal_output,
            )

        return FlavaModelOutput(
            image_embeddings=image_embeddings,
            image_output=image_output,
            text_embeddings=text_embeddings,
            text_output=text_output,
            multimodal_embeddings=multimodal_embeddings,
            multimodal_output=multimodal_output,
        )
