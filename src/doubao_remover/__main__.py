"""CLI 入口: python -m doubao_remover ..."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .pipeline import remove_watermark_file


def _resolve_prog() -> str:
    """选 argparse 显示的程序名: console script 时是 'doubao-remover',
    走 python -m 时退回到 'python -m doubao_remover'（比 '__main__.py' 友好）。
    """
    arg0 = os.path.basename(sys.argv[0] or "")
    if not arg0 or arg0 == "__main__.py":
        return "python -m doubao_remover"
    return arg0


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _same_path(a: Path, b: Path) -> bool:
    """判断两路径是否解析到同一现存文件/目录。

    用 resolve() 处理软链接和相对路径; 任何一端不存在或解析异常都视为"不同",
    走宽松路径——后续的读写操作自然会就缺失 IO 错误，无需在此处提前报错。
    """
    try:
        return a.resolve() == b.resolve()
    except (OSError, RuntimeError):
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=_resolve_prog(),
        description="豆包 AI 生图水印去除工具（基于 LaMa 深度学习模型）",
    )
    p.add_argument("-i", "--input", required=True,
                   help="输入图片路径或目录（目录则批量处理）")
    p.add_argument("-o", "--output",
                   help="输出图片路径或目录；省略时输出到 <input>_clean.<ext>，不覆盖原文件")
    p.add_argument("-t", "--template",
                   help="自定义水印模板路径（默认: assets/watermark_template.png）")
    p.add_argument("-m", "--mode", choices=("document", "photo"), default="document",
                   help="处理模式：document=保护字母细节（默认），photo=直接重画水印 bbox")
    p.add_argument("--alpha-unblend", action="store_true",
                   help="启用 alpha 反解（实验性，可能让字母 anti-alias 边变暗）")
    p.add_argument("--overwrite", action="store_true",
                   help="允许输出覆盖输入文件/目录（默认拒绝，保护原图不被误删）")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="显示详细日志")
    p.add_argument("--debug-dir",
                   help="保存中间调试图到指定目录（仅单图模式）")
    return p


def derive_default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")


def process_single(
    input_path: Path,
    output_path: Path,
    template_path: Path | None,
    debug_dir: Path | None,
    *,
    mode: str,
    alpha_unblend: bool,
    overwrite: bool,
) -> bool:
    log = logging.getLogger("cli")
    t0 = time.time()
    try:
        result = remove_watermark_file(
            input_path, output_path, template_path,
            mode=mode,
            alpha_unblend=alpha_unblend,
            debug_dir=debug_dir,
            overwrite=overwrite,
        )
    except Exception as e:
        log.error("处理失败 %s: %s", input_path, e)
        return False

    elapsed = time.time() - t0
    score_info = "fallback bbox" if result.location.is_fallback else f"score={result.location.score:.3f}"
    mask_px = (result.masks.inpaint_mask > 0).sum()
    print(f"  ✓ {input_path.name} → {output_path.name}  "
          f"[{elapsed:.1f}s, mask {mask_px} px, {score_info}]")
    return True


def process_batch(
    input_dir: Path,
    output_dir: Path,
    template_path: Path | None,
    *,
    mode: str,
    alpha_unblend: bool,
    overwrite: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ])
    if not files:
        print(f"目录中没有支持的图片: {input_dir}")
        return 0

    print(f"批量处理 {len(files)} 张图...")
    ok = 0
    for f in files:
        if process_single(f, output_dir / f.name, template_path, None,
                          mode=mode, alpha_unblend=alpha_unblend,
                          overwrite=overwrite):
            ok += 1
    print(f"\n完成 {ok}/{len(files)} 张")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"输入不存在: {input_path}", file=sys.stderr)
        return 1

    template_path = Path(args.template) if args.template else None
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    if input_path.is_dir():
        if args.output is None:
            output_dir = input_path.parent / f"{input_path.name}_clean"
        else:
            output_dir = Path(args.output)
        # 输出目录与输入目录同路径会让每张图被自身的处理结果覆盖, 默认拒绝。
        if not args.overwrite and _same_path(output_dir, input_path):
            print(
                f"拒绝把输出目录设为输入目录: {input_path}; 加 --overwrite 才允许",
                file=sys.stderr,
            )
            return 2
        if debug_dir is not None:
            print("warning: --debug-dir 在批量模式下被忽略", file=sys.stderr)
        n = process_batch(input_path, output_dir, template_path,
                          mode=args.mode, alpha_unblend=args.alpha_unblend,
                          overwrite=args.overwrite)
        return 0 if n > 0 else 1

    output_path = Path(args.output) if args.output else derive_default_output(input_path)
    # 单图模式: 防止 -o 显式指向输入文件。derive_default_output 自带 _clean 后缀,
    # 默认走的路径天然不会冲突, 这里只拦截用户主动传入相同路径的情况。
    if not args.overwrite and _same_path(output_path, input_path):
        print(
            f"拒绝把输出写到输入文件: {input_path}; 加 --overwrite 才允许",
            file=sys.stderr,
        )
        return 2
    ok = process_single(input_path, output_path, template_path, debug_dir,
                        mode=args.mode, alpha_unblend=args.alpha_unblend,
                        overwrite=args.overwrite)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
