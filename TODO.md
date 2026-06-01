# TODO

本文档作为项目推进的唯一任务清单。任务完成后，从 Active 中删除；推进过程中发现的新任务，追加到 Active。

## Active

- [ ] 接入官方 HrSegNet 实现。
  - 官方仓库：https://github.com/CHDyshli/HrSegNet4CrackSegmentation
  - 建议放在 `third_party/HrSegNet4CrackSegmentation/`，本项目自己的推理、训练适配代码放在 `scripts/`。
  - 本地网络暂时无法完成 GitHub clone；服务器上优先执行 `bash scripts/bootstrap_third_party.sh`。
- [ ] 在 RTX 4090 服务器上准备运行环境。
  - 首轮建议使用 conda：Python 3.8、PaddlePaddle GPU、PaddleSeg 2.7.0、OpenCV。
  - 环境建好后记录 CUDA、cuDNN、Paddle、PaddleSeg 版本。
- [ ] 下载官方预训练权重。
  - 先用 HrSegNet-B48 看最佳初始效果。
  - 同时保留 HrSegNet-B32 做速度/效果对比。
  - 权重放在 `weights/`，不进入 Git。
- [ ] 在服务器上验证 `scripts/infer_folder.py` 与官方 PaddleSeg 配置/权重兼容。
- [ ] 在服务器上用预训练模型跑当前数据。
- [ ] 将服务器输出同步回本地并检查可视化效果。
- [ ] 根据初步效果决定是否微调。
- [ ] 如需微调，制定标注方案。
  - 第一批建议 50-100 个代表性 tile。
  - 优先使用模型初始 mask 后人工修正，减少从零标注成本。

## Notes

- 本地无 GPU；服务器为 RTX 4090，24 GB 显存。
- 代码、配置、文档使用 Git 管理。
- 原始数据、模型权重、推理结果、训练输出不进入 Git。
- 推荐流程：本地改代码并提交，服务器拉取代码运行，结果再同步回本地查看。
