"""Locator: 模板匹配低于阈值时回退到右下角经验 bbox。"""
from __future__ import annotations

from doubao_remover.locator import locate_watermark
from doubao_remover.pipeline import _load_default_template


def test_locator_falls_back_when_score_below_threshold(test_image):
    """min_score 高到没有任何匹配能跨过去时, locator 必须走 fallback。

    设计思路: 用真模板 + 故意把阈值抬到 0.99, 保证一定走 fallback 路径;
    这样测的是 "fallback bbox 长什么样", 不依赖构造一张特殊输入图。

    (尝试过用纯灰假模板触发, 但 OpenCV TM_CCOEFF_NORMED 对零方差模板
    返回 1.0, 反而绕过 fallback。)

    检查项:
      1. is_fallback == True
      2. bbox 落在图的右下区域 (右下边界都靠近图边)
      3. bbox 完全在图内
    """
    template = _load_default_template()
    loc = locate_watermark(test_image, template, min_score=0.99)
    h, w = test_image.shape[:2]

    assert loc.is_fallback, f"应该走 fallback, 实际 score={loc.score:.3f}"

    # fallback bbox 经验位置在右下, 检查右下边界都靠近图边
    assert loc.x + loc.w > w * 0.6, (
        f"fallback bbox 应靠右, 右边界={loc.x + loc.w}, 图宽={w}"
    )
    assert loc.y + loc.h > h * 0.6, (
        f"fallback bbox 应靠下, 下边界={loc.y + loc.h}, 图高={h}"
    )

    # bbox 不能越界
    assert loc.x >= 0 and loc.x + loc.w <= w
    assert loc.y >= 0 and loc.y + loc.h <= h
