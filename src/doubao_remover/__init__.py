"""豆包 AI 生图水印去除工具。

主要模块：
    locator         多尺度模板匹配定位水印 bbox
    mask_extractor  在 bbox 内提取精细的水印掩码
    inpainter       用 LaMa 模型修复掩码区域
    pipeline        把上面三步串起来的端到端接口
"""

__version__ = "0.1.0"
