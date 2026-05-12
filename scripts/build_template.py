"""
一次性脚本：从 test_img.jpg 提取豆包水印模板。

策略（v2）：
    1. 用目测的精确坐标（test_img.jpg 上水印在右下角偏右）裁出紧 bbox
    2. 用亮度阈值 [195,235] + isGrayish 提取水印笔画
    3. 把非水印像素强制刷白，得到"白底水印"模板
    4. 保存到 assets/watermark_template.png

观察：
    - test_img.jpg 是 2048x2048
    - 水印 "豆包AI生成" 在右下角，目测位置 x ≈ 1750-2010, y ≈ 1940-2010
    - 水印宽 ~260, 高 ~70，按 2048 图宽算 logo 占 ~12.7% 宽

只在准备阶段跑一次，运行时不调用。
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGE = PROJECT_ROOT / "test_img.jpg"
OUT_TEMPLATE = PROJECT_ROOT / "assets" / "watermark_template.png"
OUT_DEBUG_DIR = PROJECT_ROOT / "assets" / "_debug"

# 目测得到的水印精确位置（基于 2048x2048 的 test_img.jpg）
# 这里裁取的范围比水印略大，让脚本能自动收敛到紧 bbox
WM_X_RANGE = (1720, 2030)
WM_Y_RANGE = (1930, 2020)


def main() -> None:
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(SRC_IMAGE))
    if img is None:
        raise SystemExit(f"无法读取 {SRC_IMAGE}")
    h, w = img.shape[:2]
    print(f"[info] 源图尺寸 {w}x{h}")

    x1, x2 = WM_X_RANGE
    y1, y2 = WM_Y_RANGE
    crop = img[y1:y2, x1:x2].copy()
    print(f"[info] 裁取范围 x[{x1}:{x2}] y[{y1}:{y2}], crop 尺寸 {crop.shape[1]}x{crop.shape[0]}")
    cv2.imwrite(str(OUT_DEBUG_DIR / "01_crop.png"), crop)

    # 提取水印像素：浅灰白色 + 亮度在 [195, 235]
    # 这个区间能避开纯白纸（>235）和黑字 anti-alias 中间灰（<195）
    b, g, r = cv2.split(crop)
    b_i, g_i, r_i = b.astype(np.int32), g.astype(np.int32), r.astype(np.int32)
    lum = (b_i + g_i + r_i) / 3.0
    is_grayish = (np.abs(b_i - g_i) < 20) & (np.abs(g_i - r_i) < 20)
    is_in_range = (lum > 195) & (lum < 235)
    wm_mask_raw = (is_grayish & is_in_range).astype(np.uint8) * 255
    cv2.imwrite(str(OUT_DEBUG_DIR / "02_wm_mask_raw.png"), wm_mask_raw)

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wm_mask = cv2.morphologyEx(wm_mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    wm_mask = cv2.morphologyEx(wm_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(str(OUT_DEBUG_DIR / "03_wm_mask_cleaned.png"), wm_mask)

    # 连通域过滤：水印笔画一般面积较小，黑字 anti-alias 残留通常更碎
    # 这里只保留 area >= 15 的（去掉极小噪点），不设上限
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wm_mask, connectivity=8)
    filtered = np.zeros_like(wm_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 15:
            filtered[labels == i] = 255
    cv2.imwrite(str(OUT_DEBUG_DIR / "04_wm_mask_filtered.png"), filtered)

    # 找包围所有水印像素的紧 bbox
    ys, xs = np.where(filtered > 0)
    if len(xs) == 0:
        raise SystemExit("没找到任何水印像素，需要调整 WM_X_RANGE/WM_Y_RANGE 或阈值")
    pad = 3
    bx1 = max(0, int(xs.min()) - pad)
    bx2 = min(crop.shape[1], int(xs.max()) + pad + 1)
    by1 = max(0, int(ys.min()) - pad)
    by2 = min(crop.shape[0], int(ys.max()) + pad + 1)
    tight_w, tight_h = bx2 - bx1, by2 - by1
    print(f"[info] 紧 bbox: x[{bx1}:{bx2}] y[{by1}:{by2}], 尺寸 {tight_w}x{tight_h}")

    # 输出"白底水印"模板：mask 为 1 处保留原像素，为 0 处刷白
    template = np.full(crop.shape, 255, dtype=np.uint8)
    template[filtered > 0] = crop[filtered > 0]
    template = template[by1:by2, bx1:bx2]

    cv2.imwrite(str(OUT_TEMPLATE), template)
    print(f"[ok] 模板已保存: {OUT_TEMPLATE} ({template.shape[1]}x{template.shape[0]})")

    # 顺便保存"形状二值模板"（笔画 = 255，背景 = 0），供 mask extraction 阶段做形状校验
    shape_mask = filtered[by1:by2, bx1:bx2]
    cv2.imwrite(str(PROJECT_ROOT / "assets" / "watermark_shape_mask.png"), shape_mask)
    print(f"[ok] 形状二值模板已保存: assets/watermark_shape_mask.png")


if __name__ == "__main__":
    main()
