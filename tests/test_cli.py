"""CLI 端到端: --overwrite 默认关时, 不允许把输出写到输入路径。

用 subprocess 调真实命令行, 保证从 argparse 到 main() 的整条路径都被覆盖。
不跑实际的 LaMa 流程 (慢, 还要下模型), 只验证早期拦截会让进程以 rc=2 退出。
"""
from __future__ import annotations

import subprocess
import sys


def test_cli_refuses_single_overwrite(test_image_path):
    """单图模式: -o 与 -i 同路径 → rc=2, 不真的跑流水线。"""
    result = subprocess.run(
        [sys.executable, "-m", "doubao_remover",
         "-i", str(test_image_path),
         "-o", str(test_image_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2, (
        f"应当 rc=2, 实际 rc={result.returncode}; stderr={result.stderr!r}"
    )
    assert "拒绝" in result.stderr, f"未见预期拒绝提示: {result.stderr!r}"


def test_cli_refuses_batch_when_output_dir_equals_input_dir(test_image_path, tmp_path):
    """批量模式: -i 是目录, -o 指向同一目录 → rc=2。

    我们准备一个 tmp 输入目录 (复制 test_img.jpg 进去),
    然后用同一目录作 -o, 期望被覆盖检查拦下来。
    """
    input_dir = tmp_path / "imgs"
    input_dir.mkdir()
    (input_dir / "a.jpg").write_bytes(test_image_path.read_bytes())

    result = subprocess.run(
        [sys.executable, "-m", "doubao_remover",
         "-i", str(input_dir),
         "-o", str(input_dir)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2, (
        f"应当 rc=2, 实际 rc={result.returncode}; stderr={result.stderr!r}"
    )
    assert "拒绝" in result.stderr, f"未见预期拒绝提示: {result.stderr!r}"
