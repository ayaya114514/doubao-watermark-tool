"""Alpha 反向解混合：从被水印叠加的像素中数学反推原始颜色。

水印的叠加模型（线性 alpha 混合）：
    I_observed = (1 - α) · I_original + α · logo

已知 α 和 logo，可解出：
    I_original = (I_observed - α · logo) / (1 - α)

用途：对水印覆盖的"重要内容"像素（如文档黑字 anti-alias 边、照片暗部细节），
不能直接 inpaint 掉，但可以反推出无水印时的颜色，保留原细节同时去除水印。

参数 α 和 logo 在这里用经验常数 (α=0.7, logo=(200,200,200))，
对豆包水印实测吻合。如果切换到其他水印工具的图，参数需要重新标定。
"""
from __future__ import annotations

import cv2
import numpy as np


DEFAULT_ALPHA = 0.7
DEFAULT_LOGO_BGR = (200, 200, 200)


def unblend(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    logo_bgr: tuple[int, int, int] = DEFAULT_LOGO_BGR,
    max_delta: float = 60.0,
) -> np.ndarray:
    """对 mask 区域内的像素做 alpha 反向解混合（带软约束）。

    软约束 (max_delta):
        反解可能把不该动的像素（如水印没盖到的字母 anti-alias 边）反向推到很深，
        视觉上字母会变得"过于实"。max_delta 限制反解结果与原像素的最大 L2 距离，
        超出时按比例衰减回退到原像素附近。

    参数:
        image_bgr: BGR ndarray
        mask: 单通道 uint8, 255 = 需要反解, 0 = 保持不变
        alpha: 水印的叠加强度 (0,1)
        logo_bgr: 水印 logo 的 BGR 颜色
        max_delta: 单个像素 BGR 三通道 L2 距离的最大允许变化，默认 60

    返回:
        反解后的 BGR ndarray（mask 外的像素与输入完全一致）
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0,1) 内，实际 {alpha}")
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(f"image/mask 尺寸不匹配: {image_bgr.shape[:2]} vs {mask.shape[:2]}")

    if mask.sum() == 0:
        return image_bgr.copy()

    observed = image_bgr.astype(np.float32)
    logo = np.array(logo_bgr, dtype=np.float32).reshape(1, 1, 3)
    raw_original = (observed - alpha * logo) / (1.0 - alpha)

    # 软约束：限制单像素的变化幅度
    delta = raw_original - observed
    delta_norm = np.linalg.norm(delta, axis=2, keepdims=True)
    scale = np.minimum(1.0, max_delta / (delta_norm + 1e-6))
    constrained = observed + delta * scale
    constrained = np.clip(constrained, 0, 255).astype(np.uint8)

    out = image_bgr.copy()
    mask_bool = mask > 127
    out[mask_bool] = constrained[mask_bool]
    return out


def estimate_logo_color(
    image_bgr: np.ndarray,
    wm_candidate_mask: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    fallback: tuple[int, int, int] = DEFAULT_LOGO_BGR,
) -> tuple[int, int, int]:
    """根据 mask 区域的观测颜色 + 假设的背景 = 255（白纸）反推 logo 颜色。

    模型：观察值 = (1-α)·255 + α·logo
    解得 logo = (观察均值 - (1-α)·255) / α

    仅在背景大概率是白纸时有效（文档场景）。如果背景不是白纸，
    估计值会偏，建议直接用 DEFAULT_LOGO_BGR。

    参数:
        image_bgr: BGR ndarray
        wm_candidate_mask: 标记水印像素的 mask (255 = 水印)
        alpha: 已知/假设的 α
        fallback: mask 为空时的回退值

    返回:
        (B, G, R) logo 颜色估计
    """
    mask_bool = wm_candidate_mask > 127
    if not mask_bool.any():
        return fallback

    # 取 mask 范围内的像素均值（按通道分别算）
    pixels = image_bgr[mask_bool].astype(np.float32)
    obs_mean = pixels.mean(axis=0)  # shape (3,)

    # 反推 logo: logo = (obs - (1-α)·255) / α
    logo = (obs_mean - (1.0 - alpha) * 255.0) / alpha
    logo = np.clip(logo, 0, 255)
    return int(logo[0]), int(logo[1]), int(logo[2])
