# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Relative-coordinate image zoom-in tool for the DeepEyes Qwen3.5 recipe.

Qwen3.5 emits bounding boxes in a normalized 0-1000 coordinate system
(relative coordinates), unlike Qwen2.5-VL which emits absolute pixel
coordinates. The plain ``ImageZoomInTool`` treats the model output as
absolute pixels, which crops the wrong region (e.g. a bbox of
[359, 202, 644, 415] on a 1000px-wide image covers less than half of the
intended area).

This tool converts the 0-1000 relative coordinates to absolute pixel
coordinates before cropping, following the verified implementation in the
case study "基于昇腾和VeRL的多模态智能体强化学习" (ImageZoomInToolRelative).

The success response text ends with "(relative: {bbox_2d})." — the reward
function in ``recipe.deepeyes.deepeyes`` checks for the literal marker
``Zoomed in on the image to the region (relative:`` to grant the tool
reward, so do not reword it.
"""

import asyncio
import base64
import io
import logging
import os
from math import ceil, floor
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Literal prefix the reward function matches on for "tool actually succeeded".
TOOL_SUCCESS_MARKER = "Zoomed in on the image to the region (relative:"


class ImageZoomInToolRelative(BaseTool):
    """Zoom in on an image using relative (0-1000) bounding-box coordinates.

    Accepts bbox_2d = [x1, y1, x2, y2] in the normalized 0-1000 system,
    converts it to absolute pixels, validates/resizes it (min dimension 28),
    crops, and returns the crop as the new observation.
    """

    # Case-study verified value (Qwen3.5); smaller than the absolute tool's 32
    # because a 0-1000 grid on small images yields finer granularity.
    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        self.timeout = float(config.get("timeout", 30))
        rate_limit = max(1, int(config.get("rate_limit", 256)))
        self._semaphore = asyncio.Semaphore(rate_limit)

    # ------------------------------------------------------------------
    # bbox helpers
    # ------------------------------------------------------------------
    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        """Validate the bounding box dimensions and aspect ratio (absolute coordinates)."""
        try:
            if not (left < right and top < bottom):
                logger.warning(f"Invalid bbox shape: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            height = bottom - top
            width = right - left

            # Prevent division by zero for zero-sized boxes
            if min(height, width) == 0:
                logger.warning(f"Bbox has zero width or height: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            if max(height, width) / min(height, width) > 100:
                logger.warning(f"Bbox aspect ratio > 100: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            return True
        except Exception as e:
            logger.warning(f"Bbox validation error: {e}")
            return False

    def _convert_relative_to_absolute(self, bbox_2d: list[float], image_width: int, image_height: int) -> list[float]:
        """Convert relative coordinates (0-1000) to absolute pixel coordinates."""
        x1, y1, x2, y2 = bbox_2d

        # Clamp relative coordinates to valid range [0, 1000]
        x1 = max(0.0, min(1000.0, float(x1)))
        y1 = max(0.0, min(1000.0, float(y1)))
        x2 = max(0.0, min(1000.0, float(x2)))
        y2 = max(0.0, min(1000.0, float(y2)))

        # Convert to absolute pixel coordinates
        abs_x1 = (x1 / 1000.0) * image_width
        abs_y1 = (y1 / 1000.0) * image_height
        abs_x2 = (x2 / 1000.0) * image_width
        abs_y2 = (y2 / 1000.0) * image_height

        return [abs_x1, abs_y1, abs_x2, abs_y2]

    def _maybe_resize_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> Optional[list[float]]:
        """Clamp, validate, and potentially resize a bounding box (absolute coordinates).

        Returns a valid bounding box, or None if validation fails.
        """
        left, top, right, bottom = bbox_2d

        # 1. Clamp the initial bounding box to the image dimensions.
        left = max(0.0, float(left))
        top = max(0.0, float(top))
        right = min(float(image_width), float(right))
        bottom = min(float(image_height), float(bottom))

        # 2. If clamped bbox is invalid, return immediately.
        if not self._validate_bbox(left, top, right, bottom):
            return None

        current_bbox = [left, top, right, bottom]
        height = bottom - top
        width = right - left

        # 3. If the box is too small, attempt to resize it from its center.
        if height < self.MIN_DIMENSION or width < self.MIN_DIMENSION:
            logger.info(f"Bbox {width}x{height} is smaller than {self.MIN_DIMENSION}, attempting resize.")
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0

            min_dim = min(height, width)
            if min_dim == 0:  # Safeguard for zero-area boxes
                return None

            ratio = self.MIN_DIMENSION / min_dim
            target_width = width * ratio
            target_height = height * ratio

            # Scale down to fit inside the image while preserving aspect ratio.
            if target_width > image_width:
                scale_down = image_width / target_width
                target_width = image_width
                target_height *= scale_down

            if target_height > image_height:
                scale_down = image_height / target_height
                target_height = image_height
                target_width *= scale_down

            new_half_width = target_width / 2.0
            new_half_height = target_height / 2.0
            new_left = center_x - new_half_width
            new_top = center_y - new_half_height

            # Shift the box to keep it inside the image boundaries.
            if new_left < 0:
                new_left = 0
            if new_top < 0:
                new_top = 0
            if new_left + target_width > image_width:
                new_left = image_width - target_width
            if new_top + target_height > image_height:
                new_top = image_height - target_height

            new_right = new_left + target_width
            new_bottom = new_top + target_height

            current_bbox = [floor(new_left), floor(new_top), ceil(new_right), ceil(new_bottom)]

        # 4. Final validation on the resulting bounding box.
        final_left, final_top, final_right, final_bottom = current_bbox
        if not self._validate_bbox(final_left, final_top, final_right, final_bottom):
            logger.warning(f"Final bbox is invalid after processing: {current_bbox}")
            return None

        final_height = floor(final_bottom) - floor(final_top)
        final_width = floor(final_right) - floor(final_left)

        if final_height < self.MIN_DIMENSION or final_width < self.MIN_DIMENSION:
            logger.warning(
                f"Final bbox size ({final_width}x{final_height}) are still smaller than minimum "
                f"({self.MIN_DIMENSION}). Original bbox: {bbox_2d}, original image size: {image_width}x{image_height}"
            )
            return None

        return current_bbox

    # ------------------------------------------------------------------
    # BaseTool API
    # ------------------------------------------------------------------
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a trajectory-scoped instance that stores the source image.

        Accepts a PIL image, raw bytes, a path, an ``http(s)://`` URL, a
        ``file://`` URI or a ``data:image/...;base64,`` URI.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        self._instance_dict[instance_id] = {"image": _load_image(image, timeout=self.timeout), "reward": 0.0}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute a zoom-in operation using relative coordinates (0-1000)."""
        bbox_2d = parameters.get("bbox_2d")
        label = parameters.get("label", "")

        if not isinstance(bbox_2d, list | tuple) or len(bbox_2d) != 4:
            return (
                ToolResponse(text="Error: bbox_2d parameter is missing or not a list of 4 numbers."),
                -0.05,
                {"success": False},
            )

        instance_data = self._instance_dict[instance_id]
        image = instance_data["image"]
        image_width, image_height = image.size

        try:
            async with self._semaphore:
                # Convert relative coordinates to absolute pixel coordinates
                absolute_bbox = self._convert_relative_to_absolute(bbox_2d, image_width, image_height)

                # Validate and potentially resize the absolute bounding box
                resized_bbox = self._maybe_resize_bbox(absolute_bbox, image_width=image_width, image_height=image_height)

                if resized_bbox is None:
                    error_msg = (
                        f"Error: The specified bounding box (relative {bbox_2d}) is invalid or results in a crop "
                        f"smaller than the minimum size of {self.MIN_DIMENSION}x{self.MIN_DIMENSION}."
                    )
                    logger.warning(f"Tool execution failed: {error_msg}")
                    return ToolResponse(text=error_msg), -0.05, {"success": False}

                cropped_image = image.crop(resized_bbox)
                logger.info(f"Cropped image size: {cropped_image.size}")
        except Exception as e:
            logger.error(f"Error processing image zoom-in: {e}")
            return ToolResponse(text=f"Error processing image zoom-in: {e}"), -0.05, {"success": False}

        # NOTE: this exact prefix is matched by compute_score for the tool reward.
        response_text = f"Zoomed in on the image to the region (relative: {bbox_2d})."
        if label:
            response_text = f"Zoomed in on the image to the region (relative: {bbox_2d}) with label {label}."

        return ToolResponse(image=[cropped_image], text=response_text), 0.0, {"success": True}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


def _load_image(image: Any, timeout: float = 30.0) -> Image.Image:
    """Resolve an image reference into an RGB ``PIL.Image``."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, dict):
        for key in ("image", "image_url"):
            if image.get(key) is not None:
                return _load_image(image[key], timeout=timeout)
        raise TypeError(f"Unsupported image dict: {sorted(image.keys())}")

    if isinstance(image, bytes | bytearray):
        return Image.open(io.BytesIO(image)).convert("RGB")

    if isinstance(image, str):
        if image.startswith("data:image/"):
            encoded = image.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        if image.startswith(("http://", "https://")):
            response = requests.get(image, timeout=timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        if image.startswith("file://"):
            image = urlparse(image).path
        return Image.open(image).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")
