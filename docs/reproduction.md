# 复现说明

## 环境记录

原实验使用 Windows + WSL2 Ubuntu 24.04、Python 3.12.3、RTX 4050 Laptop 6GB、PyTorch 2.10.0+cu126、LeRobot 0.6.2、MuJoCo 3.15.0、Gymnasium 1.4.0、Transformers 5.5.4。

完整记录位于 `requirements-recorded.txt` 和 `records/environment.json`。这些是已使用版本的记录，不是离线安装包，也不保证其他操作系统、驱动或依赖解析结果相同。已有原环境时优先复用。新建环境时，先按 PyTorch 官方安装指引准备匹配的 CUDA wheel，再安装其余记录依赖。执行时还需要 `psutil`，原实验的部分绘图工具需要 `matplotlib`。

当前入口面向 Linux / WSL2。无显示服务器时默认使用 EGL；先运行环境检查，以便发现 OpenGL、驱动或依赖问题。

## 不下载权重也能做什么

```bash
python scripts/summarize_results.py --check
python scripts/verify_release.py
python run_task.py --help
```

安装运行依赖后，还可以核对场景与状态接口：

```bash
python scripts/check_environment.py --preview
python scripts/expert_demo.py --label expert-demo
```

这些命令不启动训练，不修改封存结果。新日志写入忽略的 `runs/`。专家演示使用物块真值进行控制，不能冒充纯模型推理。

## 纯模型推理

仓库已包含最终模型的配置、处理器、tokenizer 和归一化参数。约 1.18GB 的 `model.safetensors` 没有上传。完整模型目录应含：

```text
pretrained_model/
  model.safetensors
  config.json
  training-variant.json
  policy_preprocessor.json
  policy_postprocessor.json
  policy_preprocessor_step_5_normalizer_processor.safetensors
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
  tokenizer/
```

原最终权重 SHA256：

```text
9d92ed62f2c47a3ddab11b7f86402ab257f6993a8c2c5465a5a063fa0bef123a
```

可以在 Git 忽略的 `model_bundle/033000/pretrained_model/` 中放入该权重，或直接指定现有完整目录，避免重复占用磁盘：

```bash
export TASK4_MODEL=/absolute/path/to/033000/pretrained_model
python scripts/check_environment.py --model
python run_task.py rollout --model selected --episode 32 --label model-demo
```

固定裁剪、动作时域和需要保留 FP32 的适配模块从 `training-variant.json` 读取。仓库内加载器用本地 `assets/vlm` 构建相同架构，再严格加载全部权重，不会自动从网络下载另一份模型。

## 训练需要另行提供的内容

完整示教图像 Parquet 约 11.52GB，未放入 Git。设置其根目录及任务适配前的 Base 初始化目录：

```bash
export TASK4_DATA=/absolute/path/to/lerobot_panda_pick_place_100_lowdisk
export TASK4_BASE=/absolute/path/to/base/pretrained_model
export TASK4_OUTPUT=/absolute/path/to/new-runs
python run_task.py train --name initial20k --scope formal --steps 20000 --batch-size 16 --frames-per-episode 256
```

`TASK4_DATA` 必须是原始 LeRobot v3 数据集根目录，包含对应 metadata 和图像 Parquet。Base 初始化应匹配原项目的 8 维接口和模型架构，不能将任意下载的权重文件直接改名冒充。

续训入口：

```bash
python run_task.py refine --help
```

四阶段谱系涉及不同的父权重、训练模块、优化器设置和采样计划。不能用一条“同设置训练 33000 步”代替原谱系。`records/training/` 保留训练协议，`refinement.py` 保留续训实现。复现实验需准备相应父检查点，并核对 `TASK4_REFINE_SOURCE`、`TASK4_REFINE_PARENT_STEPS` 和 `TASK4_REFINE_ADDITIONAL_STEPS`。

原先部分大体积逐帧阶段审计日志已经清理。本发布副本直接复用封存的精确锚点列表：`focused`、`lookahead`、`completion` 各有 80×256 个条目。它们与当前数据集绑定；换一套数据时需要重新构建采样计划，不能直接套用。

## 可配置路径

- `TASK4_ASSETS`：原生仿真模块根目录，默认仓库 `simulation/`。
- `TASK4_OUTPUT`：新增运行记录，默认 `runs/`。
- `TASK4_DATA`：完整训练数据根目录。
- `TASK4_EVIDENCE`：划分、归一化统计与采样计划。
- `TASK4_BASE`：任务适配前的初始化模型。
- `TASK4_MODEL`：最终模型完整目录。
- `TASK4_VLM_CONFIG`：构建 VLM 架构的小型本地配置。

所有变量均可通过 shell 的 `export` 设置。代码不会自动读取 `.env`。

## 证据边界

`results/results.jsonl` 是原 200 次实验的记录，路径元数据经过替换，数值指标不变。重新运行演示或修改配置不会自动改变这些历史结果。未上传训练图像、Base 大权重、最终大权重、全部历史候选和失败视频全集，因此这不是完整的离线训练镜像。
