"""Mask 提取: 输出 mask 必须与原图等尺寸、二值化。

pipeline 后续把 mask 当全图 mask 送 LaMa, 尺寸不一致直接抛错;
LaMa 又要求 mask 是 0/255 二值, 中间值会被强制 binarize 但意图就不清晰。
这两个不变量值得用回归测试钉住。
"""
from __future__ import annotations

import numpy as np

from doubao_remover.locator import locate_watermark
from doubao_remover.mask_extractor import extract_masks
from doubao_remover.pipeline import _load_default_template


def test_mask_shapes_match_input(test_image):
    """两个 mask 都必须和原图等高等宽, 单通道 uint8。"""
    template = _load_default_template()
    location = locate_watermark(test_image, template)
    masks = extract_masks(test_image, location)

    h, w = test_image.shape[:2]
    assert masks.inpaint_mask.shape == (h, w)
    assert masks.alpha_blend_mask.shape == (h, w)
    assert masks.inpaint_mask.dtype == np.uint8
    assert masks.alpha_blend_mask.dtype == np.uint8


def test_mask_values_are_binary(test_image):
    """mask 像素值只能是 0 或 255, LaMa 要求二值, 中间值会被静默截断。"""
    template = _load_default_template()
    location = locate_watermark(test_image, template)
    masks = extract_masks(test_image, location)

    for name, m in [("inpaint_mask", masks.inpaint_mask),
                    ("alpha_blend_mask", masks.alpha_blend_mask)]:
        unique = set(int(v) for v in np.unique(m))
        assert unique.issubset({0, 255}), f"{name} 出现非二值像素: {unique}"


def test_photo_mode_alpha_blend_mask_is_empty(test_image):
    """photo 模式下 alpha_blend 通路被关闭, alpha_blend_mask 必须全 0,
    否则会让 alpha 反解错误地处理本该交给 LaMa 重画的像素。"""
    template = _load_default_template()
    location = locate_watermark(test_image, template)
    masks = extract_masks(test_image, location, mode="photo")
    assert int(masks.alpha_blend_mask.sum()) == 0
