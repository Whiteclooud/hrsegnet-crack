# Server Setup

目标：在 RTX 4090 服务器上使用官方 HrSegNet 预训练模型，对本项目数据做滑窗推理。

## 1. 获取代码

推荐先给本地仓库配置远程 Git 仓库，然后服务器直接 clone。

```bash
git clone <your-repo-url> hrsegnet-crack
cd hrsegnet-crack
```

如果暂时还没有远程仓库，也可以先用 `rsync` 把本项目目录同步到服务器，但后续仍建议尽快切回 Git 远程仓库。

## 2. 拉取官方实现

```bash
bash scripts/bootstrap_third_party.sh
```

该脚本会把官方仓库克隆到：

```text
third_party/HrSegNet4CrackSegmentation/
```

## 3. 创建环境

先确认服务器 CUDA 版本：

```bash
nvidia-smi
```

建议使用 conda：

```bash
conda create -n hrsegnet python=3.8 -y
conda activate hrsegnet
python -m pip install -U pip
```

PaddlePaddle GPU 版本需要和服务器 CUDA 匹配。请在服务器上按 PaddlePaddle 官方安装选择器生成命令，再安装 PaddlePaddle GPU。

然后安装本项目辅助依赖：

```bash
python -m pip install -r requirements-server.txt
```

检查环境：

```bash
python scripts/check_env.py
```

## 4. 放置数据和权重

推荐目录：

```text
hrsegnet-crack/
  weights/
    hrsegnet_b48.pdparams
    hrsegnet_b32.pdparams
  data/
    *.jpeg
```

`weights/`、`data/` 不进入 Git。

如果数据在项目外部，也可以通过 `--input /path/to/data` 直接指定。

## 5. 运行推理

先用 B48 看最佳初始效果：

```bash
python scripts/infer_folder.py \
  --input data \
  --output outputs/b48_tile400_overlap96 \
  --official-repo third_party/HrSegNet4CrackSegmentation \
  --config third_party/HrSegNet4CrackSegmentation/configs/hrsegnetb48.yml \
  --weights weights/hrsegnet_b48.pdparams \
  --device gpu \
  --crop-size 400,400 \
  --overlap 96 \
  --threshold 0.5
```

再用 B32 做对比：

```bash
python scripts/infer_folder.py \
  --input data \
  --output outputs/b32_tile400_overlap96 \
  --official-repo third_party/HrSegNet4CrackSegmentation \
  --config third_party/HrSegNet4CrackSegmentation/configs/hrsegnetb32.yml \
  --weights weights/hrsegnet_b32.pdparams \
  --device gpu \
  --crop-size 400,400 \
  --overlap 96 \
  --threshold 0.5
```

输出目录包含：

- `masks/`：二值裂缝 mask
- `probs/`：裂缝概率灰度图
- `overlays/`：红色叠加图
- `previews/`：原图、mask、overlay 的横向预览图

## 6. 同步结果回本地

```bash
rsync -av user@server:/path/to/hrsegnet-crack/outputs/ outputs/
```

回本地后优先看 `previews/` 和 `overlays/`，判断是否需要微调。
