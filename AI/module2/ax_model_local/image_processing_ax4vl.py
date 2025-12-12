"""
Image processor class for Megatron-LM LLaVA.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from .configuration_ax4vl import AX4VLConfig

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    valid_images,
    make_list_of_images,
    to_numpy_array,
    validate_preprocess_arguments,
)
from transformers.image_processing_utils import BatchFeature, get_size_dict, BaseImageProcessor
from transformers.image_transforms import (
    PaddingMode,
    pad,
    to_channel_dimension_format,
)
from transformers.utils import TensorType, logging
from transformers.models.auto import AutoImageProcessor


logger = logging.get_logger(__name__)

def _get_patch_output_size(image, target_resolution):
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    return new_width, new_height

# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio

def _pad_for_patching(image, target_resolution, background_color=(0, 0, 0)):
    """
    Pad an image to a target resolution while maintaining aspect ratio.
    """
    target_width, target_height = target_resolution
    new_width, new_height = _get_patch_output_size(image, target_resolution)

    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    padded_image = Image.new(image.mode, target_resolution, background_color)
    padded_image.paste(image, (paste_x, paste_y))
    return padded_image

def _resize_for_patching(image, target_resolution):
    new_size = _get_patch_output_size(image, target_resolution)

    # Resize the image
    resized_image = image.resize(new_size)

    return resized_image

def get_target_ratios(image_size, min_num=1, max_num=6, tile_size=384):
    orig_width, orig_height = image_size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    return find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, tile_size
    )

# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L702
# Copyright (c) 2023 OpenGVLab.
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False, padding=False):
    # find the closest aspect ratio to the target
    target_aspect_ratio = get_target_ratios(image.size, min_num=min_num, max_num=max_num, tile_size=image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    if padding: # LLaVA-Next tiling strategy
        resized_img = _resize_for_patching(image, (target_width, target_height))
        resized_img = _pad_for_patching(resized_img, (target_width, target_height))
    else:   # InternVL tiling strategy
        resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class AX4VLImageProcessor(BaseImageProcessor):

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = True,
        do_tile_pad: Optional[bool] = True,
        do_convert_rgb: bool = True,
        use_thumbnail: bool = True,
        min_num_tiles: int = 1,
        max_num_tiles: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = dict(size) if size is not None else {"shortest_edge": 224}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_pad = do_pad
        self.do_tile_pad = do_tile_pad
        self.do_convert_rgb = do_convert_rgb
        self.use_thumbnail = use_thumbnail
        self.min_num_tiles = min_num_tiles
        self.max_num_tiles = max_num_tiles

    def pad(
        self,
        image: np.ndarray,
        padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
        mode: PaddingMode = PaddingMode.CONSTANT,
        constant_values: Union[float, Iterable[float]] = 0.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pads the `image` with the specified `padding` and `mode`. Padding can be in the (`height`, `width`)
        dimension of in the (`num_patches`) dimension. In the second case an iterable if tuples is expected
        as input.

        Args:
            image (`np.ndarray`):
                The image to pad.
            padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
                Padding to apply to the edges of the height, width axes. Can be one of three formats:
                - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
                - `((before, after),)` yields same before and after pad for height and width.
                - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
            mode (`PaddingMode`):
                The padding mode to use. Can be one of:
                    - `"constant"`: pads with a constant value.
                    - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                    vector along each axis.
                    - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                    - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            `np.ndarray`: The padded image.

        """

        # call the general `pad` if padding on `height/width`, otherwise it's the `num_patched` dim
        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, constant_values, data_format, input_data_format)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if mode == PaddingMode.CONSTANT:
            image = np.pad(image, padding, mode="constant", constant_values=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = np.pad(image, padding, mode="reflect")
        elif mode == PaddingMode.REPLICATE:
            image = np.pad(image, padding, mode="edge")
        elif mode == PaddingMode.SYMMETRIC:
            image = np.pad(image, padding, mode="symmetric")
        else:
            raise ValueError(f"Invalid padding mode: {mode}")
        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )
        return image

    def _pad_for_batching(
        self,
        pixel_values: List[np.ndarray],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[np.ndarray]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            List[`np.ndarray`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            self.pad(
                image,
                padding=((0, max_patch - image.shape[0]), (0, 0), (0, 0), (0, 0)),
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        images = make_list_of_images(images)

        all_images = []
        for image in images:
            if do_resize:
                image = image.resize((size["shortest_edge"], size["shortest_edge"]), resample)

            image = to_numpy_array(image)

            if input_data_format is None:
                # We assume that all images have the same channel dimension format.
                input_data_format = infer_channel_dimension_format(image)

            if is_scaled_image(image) and do_rescale:
                logger.warning_once(
                    "It looks like you are trying to rescale already rescaled images. If the input"
                    " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
                )
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            all_images.append(image)

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            for image in all_images
        ]

        return images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_batched_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        new_images, num_tiles = [], []
        image_sizes = [image.size for image in images]
        for image in images:
            if do_convert_rgb and image.mode != "RGB":
                image = image.convert("RGB")

            image_patches = dynamic_preprocess(
                image,
                min_num=self.min_num_tiles,
                max_num=self.max_num_tiles,
                image_size=self.size["shortest_edge"],
                use_thumbnail=self.use_thumbnail,
                padding=self.do_tile_pad
            )

            # preprocess patches
            pixel_values = self._preprocess(
                image_patches,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format
            )
            pixel_values = np.array(pixel_values)
            new_images.append(pixel_values)
            num_tiles.append(len(image_patches))

        if do_pad:
            processed_images = self._pad_for_batching(new_images)
        else:
            processed_images = np.concatenate(new_images)

        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes, "num_tiles": num_tiles},
            tensor_type=return_tensors
        )


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched video from {images}")

AutoImageProcessor.register(AX4VLConfig, AX4VLImageProcessor)
