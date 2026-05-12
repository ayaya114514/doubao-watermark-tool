"""Pipeline: 默认模板能从 importlib.resources 加载 + 防覆盖输入。"""
from __future__ import annotations

import pytest

from doubao_remover.pipeline import _load_default_template, remove_watermark_file


def test_default_template_loads_via_importlib():
    """随包分发的默认模板必须通过 importlib.resources 读出来,
    保证 pip install / wheel 安装后也能用 (不依赖源码目录结构)。
    """
    template = _load_default_template()
    assert template is not None
    assert template.ndim == 3, f"应为 BGR 3 通道, 实际 ndim={template.ndim}"
    assert template.shape[2] == 3
    # 已知模板尺寸 90x287; 真换模板时需要同步更新这里
    assert template.shape[0] > 0 and template.shape[1] > 0


def test_pipeline_refuses_to_overwrite_input(test_image_path):
    """默认 overwrite=False 时, 输出路径解析到输入路径必须抛 ValueError,
    防止 cv2.imwrite 先 truncate 原图再写, 一旦写失败原图就丢了。
    """
    with pytest.raises(ValueError, match="覆盖原图"):
        remove_watermark_file(test_image_path, test_image_path)
