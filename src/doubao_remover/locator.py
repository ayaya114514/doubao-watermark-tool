"""水印定位：多尺度模板匹配。

接收原图和模板，在图的右下角搜索区域内尝试多个缩放比例，
找到 matchTemplate 相关系数最高的位置。

为什么只搜右下角：
    豆包水印固定贴在右下角，缩小搜索范围既能提速，
    又能避免画面中其他位置（如左下角的灰色 logo）误匹配。

为什么多尺度：
    模板是从 2048x2048 的 test_img.jpg 提取的，但实际处理的图分辨率千变万化，
    水印物理尺寸随图缩放，必须按比例匹配。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# 模板是基于 2048 宽的图做的；新图宽度 / 2048 就是模板该缩放的基准比例
TEMPLATE_REFERENCE_IMG_WIDTH = 2048

# 搜索区域：图片右下角的多大一块？(0.5, 0.6) 表示从 50% 宽、60% 高开始到右下
SEARCH_REGION_FRACTION = (0.5, 0.6)

# 在基准比例上下采样的多尺度因子（0.7x ~ 1.4x 覆盖大部分实际情况）
SCALE_FACTORS = (0.7, 0.85, 1.0, 1.15, 1.3, 1.45)

# matchTemplate 分数阈值。实测豆包水印（淡灰色叠加在复杂背景上）的相关系数
# 普遍只在 0.3 附近，比这低的匹配往往漂移到错的位置。
# 阈值卡在 0.5 让大多数情况走 fallback bbox（经验位置稳，padding 足够）；
# 高于 0.5 才说明模板和图都很干净，匹配可信。
MIN_MATCH_SCORE = 0.5


@dataclass(frozen=True)
class WatermarkLocation:
    """水印在图上的位置。

    属性:
        x, y, w, h:        bbox（含 padding）的位置和尺寸 — 给 mask extraction
                           用，留余地避免漏检
        score:             matchTemplate 的最高相关系数
        is_fallback:       True 表示模板匹配失败，用了经验 bbox 兜底
        tight_x/y/w/h:     不含 padding 的紧 bbox — 接近水印的真实尺寸，
                           photo 模式直接拿来做矩形 mask 时用，避免重画过多
    """
    x: int
    y: int
    w: int
    h: int
    score: float
    is_fallback: bool
    tight_x: int = 0
    tight_y: int = 0
    tight_w: int = 0
    tight_h: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def tight_bbox(self) -> tuple[int, int, int, int]:
        # 兼容旧 location 对象（没有 tight_* 字段时返回 bbox）
        if self.tight_w == 0 or self.tight_h == 0:
            return self.bbox
        return self.tight_x, self.tight_y, self.tight_w, self.tight_h


def locate_watermark(
    image: np.ndarray,
    template: np.ndarray,
    *,
    search_region_fraction: tuple[float, float] = SEARCH_REGION_FRACTION,
    scale_factors: tuple[float, ...] = SCALE_FACTORS,
    min_score: float = MIN_MATCH_SCORE,
) -> WatermarkLocation:
    """在 image 上定位水印。

    参数:
        image: 原图 BGR ndarray (H, W, 3)
        template: 水印模板 BGR ndarray (h, w, 3)
        search_region_fraction: 搜索区域起点（图宽和图高的比例）。
            默认 (0.5, 0.6) 表示从 50% 宽、60% 高开始向右下搜
        scale_factors: 多尺度因子
        min_score: 低于此分数视为未找到

    返回:
        WatermarkLocation: 包含 bbox、分数、是否回退
    """
    img_h, img_w = image.shape[:2]
    tpl_h, tpl_w = template.shape[:2]

    # 搜索区域起点
    search_x0 = int(img_w * search_region_fraction[0])
    search_y0 = int(img_h * search_region_fraction[1])
    search_region = image[search_y0:img_h, search_x0:img_w]

    # 基准比例：把模板按"图宽 / 2048"缩放到当前图的"基准水印大小"
    base_scale = img_w / TEMPLATE_REFERENCE_IMG_WIDTH

    # 转灰度做匹配，对色差不敏感
    search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_score = -1.0
    best_loc: tuple[int, int] | None = None
    best_size: tuple[int, int] | None = None

    for factor in scale_factors:
        scale = base_scale * factor
        scaled_w = max(8, int(tpl_w * scale))
        scaled_h = max(8, int(tpl_h * scale))

        # 模板比搜索区还大就跳过
        if scaled_w >= search_gray.shape[1] or scaled_h >= search_gray.shape[0]:
            continue

        # 选缩放插值：放大用 LINEAR，缩小用 AREA，符合 OpenCV 建议
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h), interpolation=interp)

        result = cv2.matchTemplate(search_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = float(max_val)
            best_loc = max_loc
            best_size = (scaled_w, scaled_h)

    if best_loc is None or best_score < min_score:
        # 回退：豆包水印固定贴在右下角，相对位置经验值。
        # 实测模板尺寸 287x90 (相对 2048 图宽即 0.14*W x 0.044*W)，
        # 紧 bbox 直接按这个比例算，避免给 photo 模式留太多冗余区域。
        margin = max(8, int(img_w * 0.012))
        wm_w = max(60, int(img_w * 0.16))
        wm_h = max(20, int(img_w * 0.05))
        x = max(0, img_w - wm_w - margin)
        y = max(0, img_h - wm_h - margin)
        return _with_padding(
            WatermarkLocation(
                x=x, y=y, w=wm_w, h=wm_h,
                score=best_score if best_score > 0 else 0.0,
                is_fallback=True,
            ),
            img_w, img_h,
        )

    # 模板匹配命中：把搜索区坐标转回原图坐标
    x = search_x0 + best_loc[0]
    y = search_y0 + best_loc[1]
    w, h = best_size  # type: ignore[misc]
    return _with_padding(
        WatermarkLocation(x=x, y=y, w=w, h=h, score=best_score, is_fallback=False),
        img_w, img_h,
    )


def _with_padding(loc: WatermarkLocation, img_w: int, img_h: int) -> WatermarkLocation:
    """给 bbox 在四向上扩一点 padding，给 mask 提取留呼吸空间。

    模板匹配有时只对上水印一半，外扩 30% 能兜住偏差；
    fallback 已经偏大，再 padding 影响也不大（mask extraction 会做精细识别）。
    同时把原始紧 bbox 存进 tight_* 字段供 photo 模式使用（photo 模式直接矩形
    mask，padding 太多会让 LaMa 重画过多无关内容，要的就是紧 bbox）。
    """
    pad_w = max(8, int(loc.w * 0.15))
    pad_h = max(8, int(loc.h * 0.25))
    new_x = max(0, loc.x - pad_w)
    new_y = max(0, loc.y - pad_h)
    new_w = min(img_w - new_x, loc.w + 2 * pad_w)
    new_h = min(img_h - new_y, loc.h + 2 * pad_h)
    return WatermarkLocation(
        x=new_x, y=new_y, w=new_w, h=new_h,
        score=loc.score, is_fallback=loc.is_fallback,
        tight_x=loc.x, tight_y=loc.y, tight_w=loc.w, tight_h=loc.h,
    )


def load_template(path: str | Path) -> np.ndarray:
    """加载模板图。"""
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"无法读取模板: {path}")
    return template
