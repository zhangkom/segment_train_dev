# 中断恢复清单

每次开始或恢复工作时，按下面顺序检查。

## 1. 仓库状态

```bash
git status --short
git log --oneline -n 5
```

## 2. 文档状态

优先查看：

- [README.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/README.md)
- [docs/history.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/history.md)
- [docs/live_stream_mmseg_plan.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/live_stream_mmseg_plan.md)

## 3. 数据状态

确认目录：

```bash
ls data/live_portrait/images/train
ls data/live_portrait/masks/train
```

数据检查：

```bash
python tools/check_dataset.py --root data/live_portrait
```

## 4. 训练状态

如果已经启动训练，先检查：

```bash
ls work_dirs/live_portrait_deeplabv3plus
```

重点确认：

- 是否已有 `latest.pth`
- 最近一次训练停止在什么阶段
- 是否已经输出验证结果

## 5. 恢复动作

按优先顺序：

1. 没有数据：先准备数据
2. 数据未检查：先跑检查工具
3. 没有训练结果：执行 `bash scripts/train.sh`
4. 已有权重未评估：执行 `bash scripts/test.sh ...`
5. 评估完成后：更新 `docs/history.md`

