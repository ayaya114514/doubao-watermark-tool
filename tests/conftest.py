"""共享 fixtures: 测试图路径与加载。

约定: 真正端到端测试用仓库根的 test_img.jpg; 不存在时自动 skip,
保证克隆下来没图也能跑过 (skip != failure)。
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def test_image_path(project_root: Path) -> Path:
    """仓库根的 test_img.jpg, 用于需要真实输入的测试。"""
    path = project_root / "test_img.jpg"
    if not path.exists():
        pytest.skip(f"测试图片不存在: {path}")
    return path


@pytest.fixture
def test_image(test_image_path: Path):
    """加载 test_image_path 成 BGR ndarray。"""
    import cv2
    img = cv2.imread(str(test_image_path), cv2.IMREAD_COLOR)
    assert img is not None, f"cv2.imread 失败: {test_image_path}"
    return img
