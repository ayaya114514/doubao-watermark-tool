# 豆包水印去除工具（Python + LaMa 版）

基于深度学习模型 LaMa（Large Mask Inpainting）的豆包 AI 生图水印去除工具，
Apple Silicon 上 MPS 加速。

## 与旧版（C++/OpenCV）的区别

| 维度 | 旧版（C++） | 新版（Python） |
|---|---|---|
| 定位 | 硬编码右下角 350×150 | 多尺度模板匹配 + 经验 bbox 兜底 |
| 掩码 | 经验亮度阈值 + 无上限连通域 | 保护性掩码 + 紧 bbox 收敛 |
| 修复 | `cv::inpaint` TELEA r=5 + 魔法值回贴 | LaMa 神经网络 inpainting |
| 加速 | 单线程 OpenCV | MPS / CUDA / CPU 自动选 |
| 误伤 | 背景里灰白物体易被涂掉 | 仅在水印 bbox 内提取，误伤低 |

## 安装

```bash
python3.12 -m venv .venv   # 需要 Python 3.10-3.13（3.14 不行）
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

首次运行会下载 LaMa 模型权重（~200 MB）到 `~/.cache/torch/hub/checkpoints/`。

## 使用

```bash
# 默认（document 模式，保护文档黑字）
python -m doubao_remover -i input.jpg

# 指定输出路径
python -m doubao_remover -i input.jpg -o output.jpg

# 批量处理目录
python -m doubao_remover -i input_dir/ -o output_dir/

# photo 模式：水印彻底消除，但水印 bbox 内的细节会被 LaMa 重画
python -m doubao_remover -i input.jpg --mode photo

# 启用 alpha 反解（实验性）
python -m doubao_remover -i input.jpg --alpha-unblend

# 详细日志
python -m doubao_remover -i input.jpg -v
```

未指定 `-o` 时输出到 `<input>_clean.<ext>`，**不会覆盖原文件**。

## 模式选择

### document（默认）
- **目标**：保护文档黑字 / 暗部细节
- **代价**：水印叠在字母上的部分会有可见残留
- **适用**：截图、文档、表格、PPT、白底黑字图

### photo
- **目标**：彻底消除水印
- **代价**：水印 bbox（约 300×100 px）内的所有内容都会被 LaMa 重画，
  bbox 内的字母 / 细节会被改写
- **适用**：照片、插画、纹理背景、不在乎右下角细节是否完全一致的图

### --alpha-unblend（实验性）
对水印叠在保护区上的像素做数学反解：`I_orig = (I_obs - α·logo) / (1-α)`。
理论上能保留更多字母细节，但**实测在文档场景反而把字母 anti-alias 压暗**，
仅在你能接受字母略微变暗 / 看效果时启用。

## 性能

2048×2048 单图，Apple Silicon (M 系列)：

| Device | 单图耗时 |
|---|---|
| MPS | ~5s |
| CPU | ~280s |

模型自动选 device 优先级：MPS > CUDA > CPU。

## 目录结构

```
.
├── pyproject.toml
├── requirements.txt
├── assets/
│   ├── watermark_template.png     豆包水印参考模板（287×90）
│   └── watermark_shape_mask.png   形状二值模板（备用）
├── scripts/
│   └── build_template.py          一次性模板提取脚本
├── src/doubao_remover/
│   ├── __init__.py
│   ├── __main__.py                CLI 入口
│   ├── locator.py                 多尺度模板匹配 + 经验 bbox 兜底
│   ├── mask_extractor.py          掩码提取（document/photo 双模式）
│   ├── alpha_unblend.py           Alpha 反向解混合
│   ├── inpainter.py               LaMa 封装（含 MPS 加速）
│   └── pipeline.py                端到端串联
├── test_img.jpg                   测试样本（含水印的文档截图）
└── result_old_cpp.jpg             旧版 C++ 输出（基线对比用）
```

## 已知限制

- **document 模式残留**：水印叠在字母上的部分无法完美区分"是字母还是水印"，
  保护策略会保留这些像素，肉眼能看到淡淡的水印痕迹。
- **photo 模式重画**：bbox 内的细节都会被 LaMa 重新生成，对文档场景可能
  让字母走样。
- **依赖 Pillow<10**：被 `simple-lama-inpainting` 锁定。
- **模板单一**：当前模板从一张 2048×2048 文档图提取，跨大幅度图片
  尺寸 / 不同水印颜色变体 的泛化未充分测试。
- **未做 Vercel 部署**：当前是本地 CLI 工具。
