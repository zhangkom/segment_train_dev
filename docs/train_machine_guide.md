# 训练机执行说明

适用场景：把当前仓库拉到 Linux/CUDA 训练机后，执行首轮直播人体分割训练。

## 1. 机器建议

- Ubuntu 20.04 / 22.04
- NVIDIA GPU，显存至少 `12GB`
- CUDA 与 PyTorch 版本匹配
- Python `3.10`

## 2. 拉取仓库

```bash
git clone git@github.com:zhangkom/segment_train_dev.git
cd segment_train_dev
```

## 3. 创建环境

默认脚本只负责创建基础环境：

```bash
PYTHON_BIN=python3.10 bash scripts/setup_env.sh
source .venv/bin/activate
```

如果训练机需要特定 CUDA 版 PyTorch，先按 PyTorch 官方源安装，再补其余依赖。例如：

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 4. 准备数据

按仓库目录放置：

```text
data/live_portrait/
  images/train
  images/val
  masks/train
  masks/val
```

数据检查：

```bash
python tools/check_dataset.py --root data/live_portrait
```

## 5. 启动训练

```bash
bash scripts/train.sh
```

如需自定义输出目录：

```bash
WORK_DIR=./work_dirs/live_portrait_deeplabv3plus_v1 bash scripts/train.sh
```

## 6. 训练完成后

评估：

```bash
bash scripts/test.sh work_dirs/live_portrait_deeplabv3plus/latest.pth
```

导出：

```bash
bash scripts/export_onnx.sh work_dirs/live_portrait_deeplabv3plus/latest.pth
```

## 7. 必做回填

训练完成后必须更新：

- [docs/history.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/history.md)
- [docs/experiment_log_template.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/experiment_log_template.md)

