"""端到端 pipeline：locate → extract masks → alpha unblend → LaMa inpaint。

两层接口：
    remove_watermark_array  对 ndarray 操作
    remove_watermark_file   对文件路径操作
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .alpha_unblend import (
    DEFAULT_ALPHA,
    DEFAULT_LOGO_BGR,
    estimate_logo_color,
    unblend,
)
from .inpainter import inpaint
from .locator import WatermarkLocation, load_template, locate_watermark
from .mask_extractor import MaskBundle, extract_masks


# 默认模板的包内资源路径（相对于 doubao_remover/）。
# 用 importlib.resources 读取，是为了兼容 wheel/zipapp/editable install 等
# 各种安装形态——以前用 Path(__file__).parent.parent.parent 推导仓库根，
# 一旦走 pip install . 安装到 site-packages 就找不到 assets 目录。
DEFAULT_TEMPLATE_RESOURCE = "assets/watermark_template.png"

log = logging.getLogger(__name__)


def _load_default_template() -> np.ndarray:
    """加载随包分发的默认水印模板。

    通过 importlib.resources 读字节 + cv2.imdecode，避免硬编码文件系统路径。
    若模板缺失（package-data 配置漏了、安装损坏等），抛出明确异常。
    """
    from importlib.resources import files
    try:
        data = files(__package__).joinpath(DEFAULT_TEMPLATE_RESOURCE).read_bytes()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"未找到包内默认模板 {DEFAULT_TEMPLATE_RESOURCE}; "
            "可能 wheel 打包时漏了 package-data, 或用 -t 显式指定模板"
        ) from e
    arr = np.frombuffer(data, dtype=np.uint8)
    template = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"默认模板解码失败: {DEFAULT_TEMPLATE_RESOURCE}")
    return template


@dataclass(frozen=True)
class RemovalResult:
    """单次水印去除的产物。"""
    image: np.ndarray
    location: WatermarkLocation
    masks: MaskBundle
    estimated_logo_bgr: tuple[int, int, int]


def remove_watermark_array(
    image: np.ndarray,
    template: np.ndarray | None = None,
    *,
    mode: str = "document",
    alpha_unblend: bool = False,
    alpha: float = DEFAULT_ALPHA,
    logo_bgr: tuple[int, int, int] | None = None,
    debug_dir: str | Path | None = None,
) -> RemovalResult:
    """对单张图（ndarray）做端到端水印去除。

    参数:
        image: BGR ndarray
        template: 水印模板 ndarray, None 表示从默认路径加载
        mode: "document" 或 "photo"
            document: 保护暗部、只精细 mask 水印笔画。字母不被破坏，
                      但水印重叠字母的部分残留较多。适合文档/截图场景。
            photo:    把水印紧 bbox 整块送 LaMa 重生成。水印消除最干净，
                      但 bbox 内的暗部细节会被改写。适合照片/复杂背景场景。
        alpha_unblend: 是否对水印叠在保护区上的像素做 alpha 反解。
            默认 False（仅靠 LaMa）。开启后会数学反推水印覆盖区的原始颜色，
            理论上能保留更多字母细节，但实测对豆包水印效果不一定更好——
            字母 anti-alias 边可能被过度暗化。仅当 LaMa 结果有残留水印
            且你能接受字母略微变暗时启用。photo 模式下该开关被忽略。
        alpha: 水印叠加 α；默认 0.7（豆包水印实测值）
        logo_bgr: 水印 logo BGR 颜色；None 表示从图中自动估计
        debug_dir: 不为 None 时保存每步中间产物

    返回:
        RemovalResult
    """
    if template is None:
        template = _load_default_template()

    log.info("step 1/4: 定位水印")
    location = locate_watermark(image, template)
    log.info("  bbox=(%d,%d,%d,%d) score=%.3f fallback=%s",
             location.x, location.y, location.w, location.h,
             location.score, location.is_fallback)

    log.info("step 2/4: 提取水印掩码 (mode=%s)", mode)
    masks = extract_masks(
        image, location,
        mode=mode,
        debug_dir=str(debug_dir) if debug_dir else None,
    )
    inpaint_px = int((masks.inpaint_mask > 0).sum())
    alpha_px = int((masks.alpha_blend_mask > 0).sum())
    log.info("  inpaint_mask %d px, alpha_blend_mask %d px", inpaint_px, alpha_px)

    log.info("step 3/4: alpha 反解 (%s)", "enabled" if alpha_unblend else "skipped")
    image_after_unblend = image
    if alpha_unblend and alpha_px > 0:
        if logo_bgr is None:
            logo_bgr = estimate_logo_color(image, masks.inpaint_mask, alpha=alpha)
            log.info("  自动估计 logo_bgr=%s", logo_bgr)
        else:
            log.info("  使用指定 logo_bgr=%s", logo_bgr)
        image_after_unblend = unblend(image, masks.alpha_blend_mask,
                                       alpha=alpha, logo_bgr=logo_bgr)
        if debug_dir is not None:
            cv2.imwrite(str(Path(debug_dir) / "step_after_unblend.png"), image_after_unblend)
    final_logo_bgr = logo_bgr if logo_bgr is not None else DEFAULT_LOGO_BGR

    log.info("step 4/4: LaMa 修复")
    if inpaint_px == 0:
        log.warning("  inpaint_mask 为空, 跳过 LaMa")
        result_image = image_after_unblend
    else:
        lama_output = inpaint(image_after_unblend, masks.inpaint_mask)
        # 软边缘混合：LaMa 在矩形 mask 上常常生成"边缘突然变模糊"的过渡，
        # 用 mask 的高斯模糊版本做 alpha 通道，让 mask 中心完全用 LaMa 输出，
        # 边缘羽化过渡回原图，消除"涂抹色块的硬边"。
        blur_radius = max(11, (min(masks.inpaint_mask.shape) // 60) | 1)
        if blur_radius % 2 == 0:
            blur_radius += 1
        soft_alpha = cv2.GaussianBlur(masks.inpaint_mask, (blur_radius, blur_radius), 0)
        soft_alpha_f = (soft_alpha.astype(np.float32) / 255.0)[..., None]
        result_image = (
            lama_output.astype(np.float32) * soft_alpha_f
            + image_after_unblend.astype(np.float32) * (1.0 - soft_alpha_f)
        ).clip(0, 255).astype(np.uint8)

    if debug_dir is not None:
        cv2.imwrite(str(Path(debug_dir) / "final_result.png"), result_image)

    return RemovalResult(
        image=result_image,
        location=location,
        masks=masks,
        estimated_logo_bgr=final_logo_bgr,
    )


def remove_watermark_file(
    input_path: str | Path,
    output_path: str | Path,
    template_path: str | Path | None = None,
    *,
    mode: str = "document",
    alpha_unblend: bool = False,
    alpha: float = DEFAULT_ALPHA,
    logo_bgr: tuple[int, int, int] | None = None,
    debug_dir: str | Path | None = None,
    overwrite: bool = False,
) -> RemovalResult:
    """读文件 → 处理 → 写文件。

    新增 overwrite: 是否允许输出路径与输入路径解析到同一文件。
    默认 False, 防止误把原图覆盖丢失 (cv2.imwrite 先把文件 truncate 再写,
    原图数据会瞬间丢失)。CLI 默认也是 False, 加 --overwrite 才放行。
    """
    in_resolved = Path(input_path).resolve() if Path(input_path).exists() else None
    out = Path(output_path)
    # 防覆盖检查放在最前面, 失败也不会让 LaMa 跑完一遍才发现.
    # 只有 input 真实存在时才能 resolve 并比较, 否则按"无冲突"处理
    # (cv2.imread 后面也会就 "无法读取" 报错)。
    if in_resolved is not None and not overwrite:
        try:
            same = out.resolve() == in_resolved
        except (OSError, RuntimeError):
            same = False
        if same:
            raise ValueError(
                f"输出路径与输入路径相同, 会覆盖原图: {in_resolved}; "
                "传 overwrite=True 才允许"
            )

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")

    template = load_template(template_path) if template_path else None

    result = remove_watermark_array(
        image, template,
        mode=mode,
        alpha_unblend=alpha_unblend,
        alpha=alpha, logo_bgr=logo_bgr,
        debug_dir=debug_dir,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), result.image):
        raise IOError(f"写入失败: {output_path}")

    return result


# 兼容性别名
remove_watermark = remove_watermark_array
