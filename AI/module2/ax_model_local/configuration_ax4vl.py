import transformers
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

class LDPConfig(PretrainedConfig):
    model_type = "ldpnetv2_projector"

    def __init__(
        self,
        in_hidden_size=1024,
        out_hidden_size=2048,
        grid_size=12,
        **kwargs
    ):
        self.in_hidden_size = in_hidden_size
        self.out_hidden_size = out_hidden_size
        self.grid_size = grid_size

        super().__init__(**kwargs)

class MLPProjectorConfig(PretrainedConfig):
    model_type = "mlp2x_projector"

    def __init__(
        self,
        hidden_act="gelu",
        in_hidden_size=1024,
        out_hidden_size=2048,
        bias: bool=True,
        **kwargs
    ):
        self.hidden_act = hidden_act
        self.in_hidden_size = in_hidden_size
        self.out_hidden_size = out_hidden_size
        self.bias = bias

        super().__init__(**kwargs)



class AX4VLConfig(PretrainedConfig):
    model_type = "a.x-4-vl"
    sub_configs = {
        "text_config": AutoConfig,
        "projector_config": AutoConfig,
        "vision_config": AutoConfig
    }

    def __init__(
        self,
        vision_config=None,
        projector_config=None,
        text_config=None,
        image_token_index=102400,
        vision_feature_select_strategy="full",
        vision_feature_layer=0,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.image_token_index = image_token_index

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4304,
                hidden_size=1152,
                patch_size=16,
                image_size=384,
                num_hidden_layers=27,
                num_attention_heads=16,
                vision_use_head=False
            )
        self.vision_config = vision_config

        if isinstance(projector_config, dict):
            projector_config["model_type"] = (
                projector_config["model_type"] if "model_type" in projector_config else "mlp2x"
            )
            projector_config = CONFIG_MAPPING[projector_config["model_type"]](**projector_config)
        elif projector_config is None:
            projector_config = CONFIG_MAPPING["mlp2x_projector"]()
        self.projector_config = projector_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


AutoConfig.register(LDPConfig.model_type, LDPConfig)
AutoConfig.register(MLPProjectorConfig.model_type, MLPProjectorConfig)
AutoConfig.register(AX4VLConfig.model_type, AX4VLConfig)