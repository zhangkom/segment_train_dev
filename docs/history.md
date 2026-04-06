# 项目历史记录

## 2026-04-06

### 已完成

- 初始化空仓库
- 建立直播人体分割训练工程骨架
- 固化首版 `MMSeg` 基线方案
- 新增 `DeepLabV3+ + ResNet50` 训练配置
- 新增数据目录模板与 `.gitkeep`
- 新增环境、训练、评估、导出脚本
- 新增数据检查工具
- 新增数据规范、实施方案和恢复文档
- 新增训练机执行说明与实验记录模板
- 修正 `MMSeg` 运行入口，避免误用不存在的 `mmseg.tools.*` pip 模块
- 去掉 `openmim` 依赖，改为仓库自带 `Runner` 入口脚本，降低环境复杂度
- 本机已安装 `torch/mmcv-lite/mmengine/mmsegmentation`
- 已验证 `torch 2.2.2`、`mmcv 2.1.0`、`mmengine 0.10.7`、`mmseg 1.2.2`
- 已验证当前数据目录不存在，因此训练尚未开始

### 当前判断

- 当前最缺的是直播数据集，不是模型结构
- 先跑通高质量基线，再迁移到实时轻量模型更稳
- 当前本机更适合做文档、配置和数据检查，不适合作为正式训练机

### 下一步

- 准备 `data/live_portrait`
- 执行 `python tools/check_dataset.py --root data/live_portrait`
- 在 Linux/CUDA 训练机执行 `bash scripts/train.sh`
- 完成首轮训练后补充指标和 bad case 记录

### 备注

- 远端仓库初始状态为空仓库
- 当前尚未进行真实训练
- 本机默认 `python3` 为 `3.7.6`，已调整安装脚本为优先使用 `python3.10`
- 本地初始化改为 `mmcv-lite`，降低首轮环境搭建阻力
- 本机执行 `mmengine.runner.Runner` 导入时异常退出，后续训练建议切换到 Linux/CUDA 环境
