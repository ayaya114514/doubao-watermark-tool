"""水印掩码提取：在定位 bbox 内识别水印影响的像素。

返回两类 mask，分别对应两种处理路径：

inpaint_mask:
    送给 LaMa 生成模型重新填的区域。这些像素的原始信号被水印盖死了
    （背景是白纸或浅色），没有数学上可恢复的内容，只能靠模型"瞎编"。

alpha_blend_mask:
    送给 alpha 反解的区域。这些像素本来有重要内容（文档黑字 anti-alias 边、
    照片暗部细节），但被水印的浅灰层盖了一层。用线性反解公式
        I_original = (I_observed - α·logo) / (1-α)
    能数学反推 watermark 之前的颜色，保留原细节 + 去除水印贡献。

两个 mask 互不重叠：inpaint_mask = 水印候选 \\ 保护区，
                  alpha_blend_mask = 水印影响区 ∩ 保护区。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .locator import WatermarkLocation


@dataclass(frozen=True)
class MaskBundle:
    """提取结果。所有 mask 都是全图大小的单通道 uint8 (0/255)。"""
    inpaint_mask: np.ndarray
    alpha_blend_mask: np.ndarray


def extract_masks(
    image: np.ndarray,
    location: WatermarkLocation,
    *,
    mode: str = "document",
    brightness_low: int = 200,
    brightness_high: int = 230,
    rgb_gray_tolerance: int = 25,
    protect_threshold: int = 180,
    protect_dilate_iter: int = 2,
    influence_dilate_iter: int = 4,
    min_wm_component_area: int = 8,
    final_dilate_iter: int = 1,
    photo_mask_padding: int = 6,
    debug_dir: str | None = None,
) -> MaskBundle:
    """从 image 的 bbox 内同时提取 inpaint_mask 和 alpha_blend_mask。

    参数:
        image: BGR ndarray
        location: 由 locator 给出的水印位置
        mode: "document" | "photo"
            - document: 保护暗部（文档黑字等），只 mask 真正的水印笔画 + 影响区。
              字母不被破坏，但水印叠在字母上的部分残留较多。
            - photo: 不保护，把整个水印紧 bbox 当作 mask 让 LaMa 重生成。
              水印消除最干净，但 bbox 内的细节（包括非水印的暗部）会被 LaMa 重画。
              适合背景是照片/复杂纹理、不在意 bbox 内细节是否完全一致的场景。
        brightness_low/high: 水印像素的亮度区间（豆包水印典型 200-230）
        rgb_gray_tolerance: "近似灰色"的三通道最大差
        protect_threshold: 亮度低于此值的像素属于"必须保护"
        protect_dilate_iter: 保护区向外膨胀的迭代次数
        influence_dilate_iter: 水印候选向外膨胀以估计"水印影响区"的迭代次数
            （水印对周围 anti-alias 也有渗透，膨胀几像素能兜住）
        min_wm_component_area: 水印连通域的最小面积
        final_dilate_iter: inpaint mask 最终膨胀次数
        photo_mask_padding: photo 模式下水印紧 bbox 向外扩的像素数
        debug_dir: 不为 None 时保存中间产物

    返回:
        MaskBundle(inpaint_mask, alpha_blend_mask)
        photo 模式下 alpha_blend_mask 总为全 0（这种模式下 LaMa 接管一切）。
    """
    if mode not in ("document", "photo"):
        raise ValueError(f"mode 必须是 'document' 或 'photo', 实际 {mode!r}")
    img_h, img_w = image.shape[:2]
    x, y, w, h = location.bbox
    x2, y2 = x + w, y + h

    roi = image[y:y2, x:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. 水印颜色候选：浅灰 + 亮度在水印典型区间
    b, g, r = cv2.split(roi)
    bi, gi, ri = b.astype(np.int32), g.astype(np.int32), r.astype(np.int32)
    is_grayish = (
        (np.abs(bi - gi) < rgb_gray_tolerance) &
        (np.abs(gi - ri) < rgb_gray_tolerance) &
        (np.abs(bi - ri) < rgb_gray_tolerance)
    )
    is_in_brightness = (roi_gray > brightness_low) & (roi_gray < brightness_high)
    wm_candidate = (is_grayish & is_in_brightness).astype(np.uint8) * 255

    # 2. 保护区：暗部（黑字、阴影）+ 膨胀
    protect_raw = (roi_gray < protect_threshold).astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    protect = cv2.dilate(protect_raw, kernel3, iterations=protect_dilate_iter)

    # 3. 水印影响区：从"可信水印像素"取 5-95 百分位作为紧包围矩形。
    # 不用 min/max（少量 outlier 会把范围拉到整个 ROI），不用直接膨胀
    # （会顺着字母 anti-alias 连成一片）。百分位过滤 outlier，得到水印 dense 区。
    trusted_wm = wm_candidate.copy()
    trusted_wm[protect > 0] = 0
    wm_influence = np.zeros_like(wm_candidate)
    ys, xs = np.where(trusted_wm > 0)
    if len(xs) > 0:
        x_lo, x_hi = np.percentile(xs, [5, 95])
        y_lo, y_hi = np.percentile(ys, [5, 95])
        pad = max(8, int(0.08 * (x_hi - x_lo)))
        tx1 = max(0, int(x_lo) - pad)
        tx2 = min(wm_candidate.shape[1], int(x_hi) + pad + 1)
        ty1 = max(0, int(y_lo) - pad)
        ty2 = min(wm_candidate.shape[0], int(y_hi) + pad + 1)
        wm_influence[ty1:ty2, tx1:tx2] = 255

    # 4. inpaint_mask = 水印候选 - 保护区 → 安全送 LaMa
    inpaint_raw = wm_candidate.copy()
    inpaint_raw[protect > 0] = 0
    # 4.1 连通域过滤去孤立点
    num_wm, labels_wm, stats_wm, _ = cv2.connectedComponentsWithStats(
        inpaint_raw, connectivity=8
    )
    inpaint_filtered = np.zeros_like(inpaint_raw)
    for i in range(1, num_wm):
        if stats_wm[i, cv2.CC_STAT_AREA] >= min_wm_component_area:
            inpaint_filtered[labels_wm == i] = 255
    # 4.2 膨胀吃边 + 再次扣保护
    inpaint_expanded = cv2.dilate(inpaint_filtered, kernel3, iterations=final_dilate_iter)
    inpaint_expanded[protect > 0] = 0

    # 5. alpha_blend_mask = 水印影响区 ∩ 保护区
    #    （保护区内、水印影响范围内、原始亮度有效的像素 → 做 alpha 反解）
    alpha_blend_local = ((wm_influence > 0) & (protect > 0)).astype(np.uint8) * 255

    # 6. 模式分支
    if mode == "photo":
        # photo 模式：用 locator 的紧 bbox 作矩形 mask 送 LaMa。
        # 颜色阈值在彩色背景上失效（棕黄地面、绿草等会让 grayish 判定不命中），
        # 模板形状 mask 又因为 fallback 定位不准会漂移到错位置；折中走 tight
        # bbox 整块矩形。pipeline 后续会用高斯软边混合，让 LaMa 输出的边缘
        # 平滑过渡到原图，减少"色块硬边"。
        # 残余的"中心模糊感"是 LaMa 在大矩形 mask 上的固有特性，本算法层面
        # 无法消除，需要更强的模型（如 SD inpaint）或更精确的水印定位。
        tx, ty, tw, th = location.tight_bbox
        rel_x = max(0, tx - location.x)
        rel_y = max(0, ty - location.y)
        rel_w = min(roi.shape[1] - rel_x, tw)
        rel_h = min(roi.shape[0] - rel_y, th)
        photo_mask = np.zeros_like(wm_candidate)
        # 安全外扩像素：把 photo_mask_padding 接到此处, 让函数参数真生效。
        # 留几像素是为了兜住模板匹配/fallback bbox 的轻微偏差,
        # 不至于让水印边缘 1-2 像素漏在 LaMa 修复区外。
        safety = max(0, int(photo_mask_padding))
        sx1 = max(0, rel_x - safety)
        sy1 = max(0, rel_y - safety)
        sx2 = min(roi.shape[1], rel_x + rel_w + safety)
        sy2 = min(roi.shape[0], rel_y + rel_h + safety)
        photo_mask[sy1:sy2, sx1:sx2] = 255
        inpaint_expanded = photo_mask
        alpha_blend_local = np.zeros_like(alpha_blend_local)

    # 7. 转回全图 mask
    full_inpaint = np.zeros((img_h, img_w), dtype=np.uint8)
    full_inpaint[y:y2, x:x2] = inpaint_expanded
    full_alpha = np.zeros((img_h, img_w), dtype=np.uint8)
    full_alpha[y:y2, x:x2] = alpha_blend_local

    if debug_dir is not None:
        dbg = Path(debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "mask_01_roi.png"), roi)
        cv2.imwrite(str(dbg / "mask_02_wm_candidate.png"), wm_candidate)
        cv2.imwrite(str(dbg / "mask_03_protect.png"), protect)
        cv2.imwrite(str(dbg / "mask_04_wm_influence.png"), wm_influence)
        cv2.imwrite(str(dbg / "mask_05_inpaint_roi.png"), inpaint_expanded)
        cv2.imwrite(str(dbg / "mask_06_alpha_blend_roi.png"), alpha_blend_local)
        cv2.imwrite(str(dbg / "mask_07_full_inpaint.png"), full_inpaint)
        cv2.imwrite(str(dbg / "mask_08_full_alpha.png"), full_alpha)

    return MaskBundle(inpaint_mask=full_inpaint, alpha_blend_mask=full_alpha)


# 兼容性别名（旧的调用方式仍然能拿到 inpaint_mask）
def extract_mask(image: np.ndarray, location: WatermarkLocation, **kwargs) -> np.ndarray:
    """旧接口：返回单一 inpaint_mask。新代码请用 extract_masks()。"""
    return extract_masks(image, location, **kwargs).inpaint_mask
