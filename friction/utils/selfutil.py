
import copy
import inspect
import json
import os
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch

import PIL.Image
import PIL.ImageOps
        
ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["torch.Tensor"]
]  # noqa



class ExplicitEnum(str):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )
    

class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
    MLX = "mlx"


class ProcessorMixin():
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    attributes = ["feature_extractor", "tokenizer"]
    optional_attributes = ["chat_template"]
    optional_call_args: List[str] = []
    # Names need to be attr_class for attr in attributes
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None
    valid_kwargs: List[str] = []

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # First, extract optional attributes from kwargs if present
        # Optional attributes can never be positional arguments
        for optional_attribute in self.optional_attributes:
            setattr(self, optional_attribute, kwargs.pop(optional_attribute, None))
        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.attributes:
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.attributes):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            class_name = getattr(self, f"{attribute_name}_class")
            # Nothing is ever going to be an instance of "AutoXxx", in that case we check the base class.
            class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
            if isinstance(class_name, tuple):
                proper_class = tuple(getattr(transformers_module, n) for n in class_name if n is not None)
            else:
                proper_class = getattr(transformers_module, class_name)

            if not isinstance(arg, proper_class):
                raise TypeError(
                    f"Received a {type(arg).__name__} for argument {attribute_name}, but a {class_name} was expected."
                )

            setattr(self, attribute_name, arg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Get the kwargs in `__init__`.
        sig = inspect.signature(self.__init__)
        # Only save the attributes that are presented in the kwargs of `__init__`.
        attrs_to_save = sig.parameters
        # Don't save attributes like `tokenizer`, `image processor` etc.
        attrs_to_save = [x for x in attrs_to_save if x not in self.__class__.attributes]
        # extra attributes to be kept
        attrs_to_save += ["auto_map"]

        output = {k: v for k, v in output.items() if k in attrs_to_save}

        output["processor_class"] = self.__class__.__name__

        if "tokenizer" in output:
            del output["tokenizer"]
        if "image_processor" in output:
            del output["image_processor"]
        if "feature_extractor" in output:
            del output["feature_extractor"]
        if "chat_template" in output:
            del output["chat_template"]

        # Some attributes have different names but containing objects that are not simple strings
        output = {
            k: v
            for k, v in output.items()
            if not (isinstance(v, PushToHubMixin) or v.__class__.__name__ == "BeamSearchDecoderCTC")
        }

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.attributes]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`]. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.attributes]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
            configs.append(self)
            custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)
            attribute.save_pretrained(save_directory)

        if self._auto_class is not None:
            # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
            for attribute_name in self.attributes:
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        # If we save using the predefined names, we can load using `from_pretrained`
        # plus we save chat_template in its own file
        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_raw_chat_template_file = os.path.join(save_directory, "chat_template.jinja")
        output_chat_template_file = os.path.join(save_directory, "chat_template.json")

        processor_dict = self.to_dict()
        # Save `chat_template` in its own file. We can't get it from `processor_dict` as we popped it in `to_dict`
        # to avoid serializing chat template in json config file. So let's get it from `self` directly
        if self.chat_template is not None:
            if kwargs.get("save_raw_chat_template", False):
                with open(output_raw_chat_template_file, "w", encoding="utf-8") as writer:
                    writer.write(self.chat_template)
                logger.info(f"chat template saved in {output_raw_chat_template_file}")
            else:
                chat_template_json_string = (
                    json.dumps({"chat_template": self.chat_template}, indent=2, sort_keys=True) + "\n"
                )
                with open(output_chat_template_file, "w", encoding="utf-8") as writer:
                    writer.write(chat_template_json_string)
                logger.info(f"chat template saved in {output_chat_template_file}")

        # For now, let's not save to `processor_config.json` if the processor doesn't have extra attributes and
        # `auto_map` is not specified.
        if set(processor_dict.keys()) != {"processor_class"}:
            self.to_json_file(output_processor_file)
            logger.info(f"processor saved in {output_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        if set(processor_dict.keys()) == {"processor_class"}:
            return []
        return [output_processor_file]

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)

        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            # cant't load chat-template when given a file as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            processor_file = pretrained_model_name_or_path
            resolved_processor_file = download_url(pretrained_model_name_or_path)
            # can't load chat-template when given a file url as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
        else:
            processor_file = PROCESSOR_NAME
            chat_template_file = "chat_template.json"
            raw_chat_template_file = "chat_template.jinja"
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                # Load chat template from a separate json if exists
                # because making it part of processor-config break BC.
                # Processors in older version do not accept any kwargs
                resolved_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    chat_template_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_raw_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    raw_chat_template_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {PROCESSOR_NAME} file"
                )

        # Add chat template as kwarg before returning because most models don't have processor config
        if resolved_raw_chat_template_file is not None:
            with open(resolved_raw_chat_template_file, "r", encoding="utf-8") as reader:
                chat_template = reader.read()
            kwargs["chat_template"] = chat_template
        elif resolved_chat_template_file is not None:
            with open(resolved_chat_template_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            chat_template = json.loads(text)["chat_template"]
            kwargs["chat_template"] = chat_template

        # Existing processors on the Hub created before #27761 being merged don't have `processor_config.json` (if not
        # updated afterward), and we need to keep `from_pretrained` work. So here it fallbacks to the empty dict.
        # (`cached_file` called using `_raise_exceptions_for_missing_entries=False` to avoid exception)
        # However, for models added in the future, we won't get the expected error if this file is missing.
        if resolved_processor_file is None:
            return {}, kwargs

        try:
            # Load processor dict
            with open(resolved_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        if "chat_template" in processor_dict and processor_dict["chat_template"] is not None:
            logger.warning_once(
                "Chat templates should be in a 'chat_template.jinja' file but found key='chat_template' "
                "in the processor's config. Make sure to move your template to its own file."
            )

        if not is_local:
            if "auto_map" in processor_dict:
                processor_dict["auto_map"] = add_model_info_to_auto_map(
                    processor_dict["auto_map"], pretrained_model_name_or_path
                )
            if "custom_pipelines" in processor_dict:
                processor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                    processor_dict["custom_pipelines"], pretrained_model_name_or_path
                )

        return processor_dict, kwargs

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        chat_template = kwargs.pop("chat_template", None)

        # We have to pop up some unused (but specific) kwargs and then validate that it doesn't contain unused kwargs
        # If we don't pop, some specific kwargs will raise a warning
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        if "auto_map" in processor_dict:
            del processor_dict["auto_map"]

        unused_kwargs = cls.validate_init_kwargs(processor_config=processor_dict, valid_kwargs=cls.valid_kwargs)
        processor = cls(*args, **processor_dict)
        if chat_template is not None:
            setattr(processor, "chat_template", chat_template)

        # Update processor with kwargs if needed
        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        kwargs.update(unused_kwargs)
        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        tokenizer_init_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Dict]:
        """
        Method to merge dictionaries of kwargs cleanly separated by modality within a Processor instance.
        The order of operations is as follows:
            1) kwargs passed as before have highest priority to preserve BC.
                ```python
                high_priority_kwargs = {"crop_size" = {"height": 222, "width": 222}, "padding" = "max_length"}
                processor(..., **high_priority_kwargs)
                ```
            2) kwargs passed as modality-specific kwargs have second priority. This is the recommended API.
                ```python
                processor(..., text_kwargs={"padding": "max_length"}, images_kwargs={"crop_size": {"height": 222, "width": 222}}})
                ```
            3) kwargs passed during instantiation of a modality processor have fourth priority.
                ```python
                tokenizer = tokenizer_class(..., {"padding": "max_length"})
                image_processor = image_processor_class(...)
                processor(tokenizer, image_processor) # will pass max_length unless overriden by kwargs at call
                ```
            4) defaults kwargs specified at processor level have lowest priority.
                ```python
                class MyProcessingKwargs(ProcessingKwargs, CommonKwargs, TextKwargs, ImagesKwargs, total=False):
                    _defaults = {
                        "text_kwargs": {
                            "padding": "max_length",
                            "max_length": 64,
                        },
                    }
                ```
        Args:
            ModelProcessorKwargs (`ProcessingKwargs`):
                Typed dictionary of kwargs specifically required by the model passed.
            tokenizer_init_kwargs (`Dict`, *optional*):
                Dictionary of kwargs the tokenizer was instantiated with and need to take precedence over defaults.

        Returns:
            output_kwargs (`Dict`):
                Dictionary of per-modality kwargs to be passed to each modality-specific processor.

        """
        # Initialize dictionaries
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        used_keys = set()

        # get defaults from set model processor kwargs if they exist
        for modality in default_kwargs:
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            # update defaults with arguments from tokenizer init
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                # init with tokenizer init kwargs if necessary
                if modality_key in tokenizer_init_kwargs:
                    value = (
                        getattr(self.tokenizer, modality_key)
                        if hasattr(self.tokenizer, modality_key)
                        else tokenizer_init_kwargs[modality_key]
                    )
                    default_kwargs[modality][modality_key] = value
        # now defaults kwargs are updated with the tokenizers defaults.
        # pass defaults to output dictionary
        output_kwargs.update(default_kwargs)

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality in output_kwargs:
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                # check if we received a structured kwarg dict or not to handle it correctly
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    # check if this key was passed as a flat kwarg.
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                elif modality_key in kwargs:
                    # we get a modality_key instead of popping it because modality-specific processors
                    # can have overlapping kwargs
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if kwarg_value != "__empty__":
                    output_kwargs[modality][modality_key] = kwarg_value
                    used_keys.add(modality_key)

        # Determine if kwargs is a flat dictionary or contains nested dictionaries
        if any(key in default_kwargs for key in kwargs):
            # kwargs is dictionary-based, and some keys match modality names
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            # kwargs is a flat dictionary
            for key in kwargs:
                if key not in used_keys:
                    if key in ModelProcessorKwargs.__annotations__["common_kwargs"].__annotations__.keys():
                        output_kwargs["common_kwargs"][key] = kwargs[key]
                    else:
                        logger.warning_once(
                            f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                        )

        # all modality-specific kwargs are updated with common kwargs
        for modality in output_kwargs:
            output_kwargs[modality].update(output_kwargs["common_kwargs"])
        return output_kwargs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_args_and_dict(args, processor_dict, **kwargs)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = getattr(transformers_module, class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args

    @property
    def model_input_names(self):
        first_attribute = getattr(self, self.attributes[0])
        return getattr(first_attribute, "model_input_names", None)

    @staticmethod
    def validate_init_kwargs(processor_config, valid_kwargs):
        kwargs_from_config = processor_config.keys()
        unused_kwargs = {}
        unused_keys = set(kwargs_from_config) - set(valid_kwargs)
        if unused_keys:
            unused_key_str = ", ".join(unused_keys)
            logger.warning(
                f"Some kwargs in processor config are unused and will not have any effect: {unused_key_str}. "
            )
            unused_kwargs = {k: processor_config[k] for k in unused_keys}
        return unused_kwargs

    def prepare_and_validate_optional_call_args(self, *args):
        """
        Matches optional positional arguments to their corresponding names in `optional_call_args`
        in the processor class in the order they are passed to the processor call.

        Note that this should only be used in the `__call__` method of the processors with special
        arguments. Special arguments are arguments that aren't `text`, `images`, `audio`, nor `videos`
        but also aren't passed to the tokenizer, image processor, etc. Examples of such processors are:
            - `CLIPSegProcessor`
            - `LayoutLMv2Processor`
            - `OwlViTProcessor`

        Also note that passing by position to the processor call is now deprecated and will be disallowed
        in future versions. We only have this for backward compatibility.

        Example:
            Suppose that the processor class has `optional_call_args = ["arg_name_1", "arg_name_2"]`.
            And we define the call method as:
            ```python
            def __call__(
                self,
                text: str,
                images: Optional[ImageInput] = None,
                *arg,
                audio=None,
                videos=None,
            )
            ```

            Then, if we call the processor as:
            ```python
            images = [...]
            processor("What is common in these images?", images, arg_value_1, arg_value_2)
            ```

            Then, this method will return:
            ```python
            {
                "arg_name_1": arg_value_1,
                "arg_name_2": arg_value_2,
            }
            ```
            which we could then pass as kwargs to `self._merge_kwargs`
        """
        if len(args):
            warnings.warn(
                "Passing positional arguments to the processor call is now deprecated and will be disallowed in v4.47. "
                "Please pass all arguments as keyword arguments."
            )
        if len(args) > len(self.optional_call_args):
            raise ValueError(
                f"Expected *at most* {len(self.optional_call_args)} optional positional arguments in processor call"
                f"which will be matched with {' '.join(self.optional_call_args)} in the order they are passed."
                f"However, got {len(args)} positional arguments instead."
                "Please pass all arguments as keyword arguments instead (e.g. `processor(arg_name_1=..., arg_name_2=...))`."
            )
        return {arg_name: arg_value for arg_value, arg_name in zip(args, self.optional_call_args)}

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]]],
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
            tokenize (`bool`, *optional*, defaults to `False`):
                Whether to tokenize the output or not.
            **kwargs:
                Additional keyword arguments
        """

        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument. See "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
                )
        return self.tokenizer.apply_chat_template(
            conversation, chat_template=chat_template, tokenize=tokenize, **kwargs
        )

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)


class BatchEncoding(UserDict):
    """
    Holds the output of the [`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`],
    [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`] and
    [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
            ('input_ids', 'attention_mask', etc.).
        encoding (`tokenizers.Encoding` or `Sequence[tokenizers.Encoding]`, *optional*):
            If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
            space to token space the `tokenizers.Encoding` instance or list of instance (for batches) hold this
            information.
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above). Note that this
            parameter has an effect if the parameter `tensor_type` is set, *otherwise has no effect*.
        n_sequences (`Optional[int]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data)

        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self) -> Optional[int]:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        return self._n_sequences

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Indicate whether this [`BatchEncoding`] was generated from the result of a [`PreTrainedTokenizerFast`]
        or not.
        """
        return self._encodings is not None

    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `tokenizers.Encoding` for batch item with index `key`.

        If the key is a slice, returns the value of the dict associated to `key` ('input_ids', 'attention_mask', etc.)
        with the constraint of slice.
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        elif isinstance(item, slice):
            return {key: self.data[key][item] for key in self.data.keys()}
        else:
            raise KeyError(
                "Invalid key. Only three types of key are available: "
                "(1) string, (2) integers for backend Encoding, and (3) slices for data subsetting."
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        """
        `Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    def tokens(self, batch_index: int = 0) -> List[str]:
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[str]`: The list of tokens at that index.
        """
        if not self._encodings:
            raise ValueError(
                "tokens() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].tokens

    def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to the id of their original sentences:

            - `None` for special tokens added around or between sequences,
            - `0` for tokens corresponding to words in the first sequence,
            - `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
              encoded.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the sequence id corresponding to each token. Special tokens added
            by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding
            sequence.
        """
        if not self._encodings:
            raise ValueError(
                "sequence_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].sequence_ids

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        if not self._encodings:
            raise ValueError(
                "words() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        warnings.warn(
            "`BatchEncoding.words()` property is deprecated and should be replaced with the identical, "
            "but more self-explanatory `BatchEncoding.word_ids()` property.",
            FutureWarning,
        )
        return self.word_ids(batch_index)

    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        if not self._encodings:
            raise ValueError(
                "word_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].word_ids

    def token_to_sequence(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the sequence represented by the given token. In the general use case, this method returns `0`
        for a single sequence or the first sequence of a pair, and `1` for the second sequence of a pair

        Can be called as:

        - `self.token_to_sequence(token_index)` if batch size is 1
        - `self.token_to_sequence(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError("token_to_sequence() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_sequence(token_index)

    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

        Can be called as:

        - `self.token_to_word(token_index)` if batch size is 1
        - `self.token_to_word(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> Optional[TokenSpan]:
        """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a [`~tokenization_utils_base.TokenSpan`] with:

        - **start** -- Index of the first token.
        - **end** -- Index of the token following the last token.

        Can be called as:

        - `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
        - `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to
          1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the word in the sequence.
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            ([`~tokenization_utils_base.TokenSpan`], *optional*): Span of tokens in the encoded sequence. Returns
            `None` if no tokens correspond to the word. This can happen especially when the token is a special token
            that has been used to format the tokenization. For example when we add a class token at the very beginning
            of the tokenization.
        """

        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)
        return TokenSpan(*span) if span is not None else None

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a [`~tokenization_utils_base.CharSpan`] with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - `self.token_to_chars(token_index)` if batch size is 1
        - `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token or tokens in
                the sequence.

        Returns:
            [`~tokenization_utils_base.CharSpan`]: Span of characters in the original string, or None, if the token
            (e.g. <s>, </s>) doesn't correspond to any chars in the origin string.
        """

        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        span_indices = self._encodings[batch_index].token_to_chars(token_index)

        return CharSpan(*span_indices) if span_indices is not None else None

    def char_to_token(
        self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0
    ) -> int:
        """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - `self.char_to_token(char_index)` if batch size is 1
        - `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int`: Index of the token, or None if the char index refers to a whitespace only token and whitespace is
                   trimmed with `trim_offsets=True`.
        """

        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index, sequence_index)

    def word_to_chars(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> CharSpan:
        """
        Get the character span in the original string corresponding to given word in a sequence of the batch.

        Character spans are returned as a CharSpan NamedTuple with:

        - start: index of the first character in the original string
        - end: index of the character following the last character in the original string

        Can be called as:

        - `self.word_to_chars(word_index)` if batch size is 1
        - `self.word_to_chars(batch_index, word_index)` if batch size is greater or equal to 1

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            `CharSpan` or `List[CharSpan]`: Span(s) of the associated character or characters in the string. CharSpan
            are NamedTuple with:

                - start: index of the first character associated to the token in the original string
                - end: index of the character following the last character associated to the token in the original
                  string
        """

        if not self._encodings:
            raise ValueError("word_to_chars() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index, sequence_index)))

    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0) -> int:
        """
        Get the word in the original string corresponding to a character in the original string of a sequence of the
        batch.

        Can be called as:

        - `self.char_to_word(char_index)` if batch size is 1
        - `self.char_to_word(batch_index, char_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the character in the original string.
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the character in the
                original string.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int` or `List[int]`: Index or indices of the associated encoded token(s).
        """

        if not self._encodings:
            raise ValueError("char_to_word() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index, sequence_index)

    def convert_to_tensors(
        self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            is_tensor = torch.is_tensor

            def as_tensor(value, dtype=None):
                if isinstance(value, list) and isinstance(value[0], np.ndarray):
                    return torch.from_numpy(np.array(value))
                return torch.tensor(value)

        elif tensor_type == TensorType.JAX:
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            as_tensor = jnp.array
            is_tensor = is_jax_tensor

        elif tensor_type == TensorType.MLX:
            if not is_mlx_available():
                raise ImportError("Unable to convert output to MLX tensors format, MLX is not installed.")
            import mlx.core as mx

            as_tensor = mx.array

            def is_tensor(obj):
                return isinstance(obj, mx.array)
        else:

            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                    # # at-least2d
                    # if tensor.ndim > 2:
                    #     tensor = tensor.squeeze(0)
                    # elif tensor.ndim < 2:
                    #     tensor = tensor[None, :]

                    self[key] = tensor
            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return self

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.

        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self
