# 直播人体分割 MMSeg 实施方案

## 1. 项目目标

训练一个面向直播场景的人体前景分割模型，优先满足：

- 单人直播间主体分割稳定
- 头发、肩部、手臂边界可接受
- 可作为后续实时轻量模型的教师或基线

当前首版目标不是直接追求端侧最优时延，而是先建立：

- 可复现的数据与训练管线
- 可量化的评估指标
- 可中断续做的实施记录

## 2. 技术路线

### 当前选择

- 框架：`PyTorch + MMSegmentation`
- 基线模型：`DeepLabV3+ + ResNet50`
- 分割类别：`background / person`
- 输入尺寸：`512x512`

### 选择原因

- `DeepLabV3+` 稳定，适合做高质量基线
- 对直播场景，先把边界质量和数据规范跑通，比一开始追求极致时延更稳
- 后续可以迁移到 `PIDNet-S` 或 `BiSeNetV2` 做实时版

## 3. 数据策略

### 公开数据阶段

使用公开人体分割数据做预热或混合训练：

- Supervisely Person
- COCO 中的 `person` mask
- CIHP / ATR 仅在需要扩展到人体解析时使用

### 直播数据阶段

优先补齐以下场景：

- 半身主播
- 全身站播
- 暗光直播间
- RGB 补光灯强曝光
- 绿色幕布与普通背景混用
- 手势频繁变化
- 道具遮挡
- 发丝边缘

### 数据规模建议

- 可用版本：`5k - 10k`
- 稳定版本：`20k+`
- 难例专项：单独维护 `hard_cases`

## 4. 标注规则

统一按可见区域做 mask，当前约束如下：

- `1` 包含头发、脸、颈部、身体、四肢
- 帽子默认算前景，仅当需要严格人体本体时再单独修规则
- 手持麦克风、手机、提词器默认不算人体
- 镂空区要扣空
- 多人直播时，所有人统一记作 `person`
- 大面积运动模糊按可见主体轮廓标注

详细规范见 [docs/data_spec.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/data_spec.md)。

## 5. 训练计划

### Phase 1

- 整理 `train / val`
- 执行数据检查
- 训练 `DeepLabV3+`
- 输出首轮指标和 bad case

### Phase 2

- 补齐 bad case 数据
- 调整增强和采样策略
- 做直播场景专项评估

### Phase 3

- 训练轻量模型
- 用基线模型或伪标签做蒸馏
- 导出 ONNX / TensorRT

## 6. 评估口径

至少记录以下内容：

- `mIoU`
- `mDice`
- 验证集误检率和漏检率
- 发丝、肩部、手部边界主观检查
- 单卡吞吐
- ONNX 导出是否成功

## 7. 工程约定

- 数据不直接提交到 Git
- 每次实验必须更新 [docs/history.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/history.md)
- 中断前必须更新 [docs/resume_checklist.md](/Users/kom/work_home/workspace_ai/workspace_seg_train/docs/resume_checklist.md)
- 新实验目录统一落在 `work_dirs/`

## 8. 下一步

仓库当前已完成基础骨架，接下来按顺序执行：

1. 准备 `data/live_portrait`
2. 执行数据检查脚本
3. 启动首轮 baseline 训练
4. 回填指标到历史记录

