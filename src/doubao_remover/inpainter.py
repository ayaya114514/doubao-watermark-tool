"""LaMa 神经网络修复：把 mask 区域用周围信息生成合理内容。

封装了 simple_lama_inpainting 的细节，提供一个简单接口：
    inpaint(image_bgr, mask_uint8) -> image_bgr

LaMa 论文: 《Resolution-robust Large Mask Inpainting with Fourier Convolutions》WACV 2022
作者实现: https://github.com/advimman/lama
本封装: https://pypi.org/project/simple-lama-inpainting/

模型权重首次运行时由 simple_lama 自动下载到 ~/.cache/torch/hub/...，约 200MB。

注意:
    - simple_lama_inpainting 内部用 PyTorch CPU 推理，单图 1024px 约 2-5s
    - 它接收 PIL Image（RGB），我们做 BGR/PIL 转换
    - 它要求 mask 是单通道，值为 0 或 255
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


_lama_instance = None
_lama_device: str | None = None


def _select_device() -> "torch.device":
    """优先 MPS > CUDA > CPU。

    simple-lama-inpainting 默认只检测 cuda，没考虑 Apple Silicon 的 MPS，
    导致在 Mac 上跑成纯 CPU（2048px 单图 ~5 分钟）。
    显式传 MPS 后能压到 ~10s 量级。
    """
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_lama():
    """惰性加载 LaMa 模型（首次调用 ~3s）。"""
    global _lama_instance, _lama_device
    if _lama_instance is None:
        # 把这个 import 放到函数里，避免在没 mask 要修复时也加载 torch
        from simple_lama_inpainting import SimpleLama
        device = _select_device()
        _lama_instance = SimpleLama(device=device)
        _lama_device = str(device)
    return _lama_instance


def get_active_device() -> str | None:
    """返回当前 LaMa 所在的 device 名（懒加载，调用前可能是 None）。"""
    return _lama_device


def inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """对 image 的 mask 区域做 LaMa 修复。

    参数:
        image_bgr: BGR ndarray (H, W, 3)
        mask: 单通道 uint8 ndarray (H, W), 255 = 要修复, 0 = 保留

    返回:
        修复后的 BGR ndarray (H, W, 3)
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr 应为 (H,W,3) BGR, 实际 {image_bgr.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask 应为 (H,W) 单通道, 实际 {mask.shape}")
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError(f"mask 和 image 尺寸不匹配: {mask.shape} vs {image_bgr.shape[:2]}")

    # mask 像素值不严格是 0/255 时，二值化一下（>127 视为修复区）
    binary_mask = (mask > 127).astype(np.uint8) * 255

    # 没有任何要修复的像素：直接返回原图
    if binary_mask.sum() == 0:
        return image_bgr.copy()

    lama = _get_lama()

    # simple_lama 要 PIL RGB image 和 PIL L mask
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(binary_mask, mode="L")

    result_pil = lama(pil_image, pil_mask)

    # 结果转回 BGR ndarray
    result_rgb = np.array(result_pil)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
