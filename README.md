# HrSegNet Crack Segmentation

本项目用于复现 HrSegNet，并在本地 DJI 巡检图像上使用官方预训练模型进行裂缝分割推理，后续根据效果决定是否微调。

## 推荐工作流

本地负责代码、配置和文档管理；RTX 4090 服务器负责推理和训练。

```bash
# local
git status
git add .
git commit -m "Update HrSegNet reproduction project"
git push

# server
git pull
python scripts/infer_folder.py --input ../data --weights weights/hrsegnet_b48.pdparams --output outputs/b48

# local, after server run
rsync -av user@server:/path/to/hrsegnet-crack/outputs/ outputs/
```

## 目录约定

```text
hrsegnet-crack/
  README.md
  TODO.md
  configs/
  scripts/
  third_party/
  weights/      # 不进入 Git
  outputs/      # 不进入 Git
```

当前原始数据位于父目录的 `../data/`。推理脚本应支持通过参数传入任意数据目录。

## Git 策略

进入 Git：

- 项目脚本
- 配置文件
- 环境说明
- 文档
- 小型实验元数据

不进入 Git：

- 原始图像数据
- 预训练权重
- checkpoint
- TensorRT engine
- 推理 mask、overlay、可视化结果
- 训练日志和中间输出

## 当前里程碑

第一阶段目标是：使用官方预训练 HrSegNet-B48/B32，在 `../data/` 中的 DJI 原图上完成滑窗推理，并生成可视化 overlay，用于判断迁移效果。当前默认推理阈值为 `0.4`。

任务推进以 `TODO.md` 为准。
