# Live Portrait Segmentation Training

面向直播场景的人体前景分割训练工程，采用 `PyTorch + MMSegmentation` 作为首版训练方案。

当前仓库目标：

- 快速跑通直播场景 `person / background` 二分类分割基线
- 固化数据规范、训练流程、评估口径和导出步骤
- 保留可中断续做的实施记录，避免项目中途丢失上下文

## 1. 推荐路线

第一阶段先做稳定基线：

- 模型：`DeepLabV3+ + ResNet50`
- 输入：`512x512`
- 任务：直播人体前景二分类分割
- 损失：`CrossEntropy + Dice`
- 指标：`mIoU`、`mDice`、边界目检

第二阶段再做实时优化：

- 轻量模型迁移到 `PIDNet-S` 或 `BiSeNetV2`
- 做蒸馏、量化和 TensorRT / NCNN 导出

## 2. 目录结构

```text
configs/live_portrait/           MMSeg 配置
docs/                            方案、规范、历史记录
scripts/                         环境、训练、评估、导出脚本
tools/                           数据检查工具
requirements.txt                 Python 依赖
```

## 3. 快速开始

### 3.1 创建环境

```bash
bash scripts/setup_env.sh
```

默认使用 `python3.10`。如需指定解释器：

```bash
PYTHON_BIN=/usr/local/bin/python3.10 bash scripts/setup_env.sh
```

### 3.2 准备数据

数据目录默认约定：

```text
data/live_portrait/
  images/
    train/
    val/
  masks/
    train/
    val/
```

其中：

- 图片支持 `jpg/png/jpeg`
- mask 为单通道 PNG
- `0` 表示背景
- `1` 表示人体前景

先执行一次检查：

```bash
python tools/check_dataset.py --root data/live_portrait
```

### 3.3 开始训练

```bash
bash scripts/train.sh
```

### 3.4 模型评估

```bash
bash scripts/test.sh work_dirs/live_portrait_deeplabv3plus/latest.pth
```

### 3.5 导出 ONNX

```bash
bash scripts/export_onnx.sh work_dirs/live_portrait_deeplabv3plus/latest.pth
```

## 4. 文档入口

- 方案说明：[docs/live_stream_mmseg_plan.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/live_stream_mmseg_plan.md)
- 数据规范：[docs/data_spec.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/data_spec.md)
- 项目历史：[docs/history.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/history.md)
- 中断恢复：[docs/resume_checklist.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/resume_checklist.md)

## 5. 当前状态

当前仓库已完成：

- 直播人体分割首版方案落地
- 基线训练配置初始化
- 数据检查脚本初始化
- 训练、评估、导出入口全部改为仓库本地脚本
- 历史记录与恢复文档初始化

未完成：

- 数据集入库
- 首轮训练
- 首轮评估报告
- 导出和推理基准

## 6. 环境说明

- 推荐训练环境：`Linux + CUDA`
- 本仓库本地初始化默认使用 `mmcv-lite`
- 如果后续在 GPU Linux 机器上训练，需要按目标 CUDA 版本补装对应 `torch` 轮子
- 当前这台本机已完成依赖安装与基础导入验证，但 `mmengine.runner.Runner` 导入异常，训练不建议在本机继续跑
