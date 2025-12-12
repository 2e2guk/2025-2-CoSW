from typing import List, Union
from .configuration_ax4vl import AX4VLConfig
from transformers.models.auto import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
try:
    from transformers.processing_utils import _validate_images_text_input_order
except Exception:
    def _validate_images_text_input_order(images=None, text=None, **kwargs):
        return images, text




class BaseAXProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"


class AX4VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_pad": False,
        },
    }


class AX4VLProcessor(BaseAXProcessor):
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "num_tokens_per_tile",
        "image_token",
    ]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=16,
        num_tokens_per_tile=144,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        chat_template=None,
        **kwargs
    ):
        self.patch_size = patch_size
        self.num_tokens_per_tile = num_tokens_per_tile
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        conversations: List = None,
        **kwargs
    ) -> BatchFeature:
        if images is None and conversations is None and text is None:
            raise ValueError("You have to specify at least images, text or conversation.")

        if not text and conversations is not None:
            if isinstance(conversations[0], dict):
                conversations = [conversations]
            text = [self.apply_chat_template(conv, **kwargs) for conv in conversations]

        images, text = _validate_images_text_input_order(images, text)
        
        output_kwargs = self._merge_kwargs(
            AX4VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        prompt_strings = text
        if image_inputs:
            num_tiles = iter(image_inputs["num_tiles"])
            prompt_strings = []
            for sample in text:
                while self.image_token in sample:
                    num_tile = next(num_tiles)
                    num_image_tokens = num_tile * self.num_tokens_per_tile
                    sample = sample.replace(self.image_token, "<placeholder>" * num_image_tokens, 1)
                prompt_strings.append(sample)
            prompt_strings = [sample.replace("<placeholder>", self.image_token) for sample in prompt_strings]

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        
        if "num_tiles" in image_inputs:
            del image_inputs["num_tiles"]
        return BatchFeature(data={**text_inputs, **image_inputs})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


AutoProcessor.register(AX4VLConfig, AX4VLProcessor)
