# DeepEyes GRPO 从零搭建新环境（存档备查）

> 维护人：archted ｜ 存档：2026-11-10
> ⚠️ **本文是存档**：正常复现请用一体镜像 `deepeyes-train-infer:v1`（见 `ARCHTED_CASE.md`，更快更稳）。
> 本文档记录"不用一体镜像、从基础镜像手动搭出全量环境"的完整路线——仅在需要重构镜像内容、
> 或换基础镜像版本时才需要照做。

---

## 1. 与一体镜像的关系

一体镜像 = 本文的手动步骤 + `quay.io/ascend/vllm-ascend:v0.27.1rc1` 基线，一次性固化而成。
即：本文 §2–§5 做完后得到的环境 ≈ 一体镜像内已固化的内容。若未来软件版本升级
（如 transformers / verl 换版），先按本文路线在容器里验证，再按 CASE §9 重新构建镜像。

---

## 2. 起基础容器

```bash
docker run -d --name deepeyes --net=host --shm-size=64g --ulimit memlock=-1:-1 \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /home/h00516332/deepeyes_run:/workspace -w /workspace \
  quay.io/ascend/vllm-ascend:v0.27.1rc1 bash -lc "sleep infinity"
```

要点：`--net=host`（judge 跨机调用 + HCCL 端口）；9 个 davinci 设备一个不能少；
宿主机驱动目录 bind mount 进来（容器内 CANN 与宿主机驱动版本要兼容）。

## 3. 容器内环境调整（镜像自带 env 基础上）

```bash
docker exec -it deepeyes bash
# ① transformers 降级（镜像自带 5.14.1 与 verl(<5.13) 的 gemini3 类 import 不兼容）
/workspace/conda/verl/bin/pip install transformers==5.12.1
# ② 数据预处理需要 qwen-vl-utils（不能带依赖，会拉坏 torch 栈）
/workspace/conda/verl/bin/pip install qwen-vl-utils==0.0.14 --no-deps
```

> 环境耦合原理（2026-11-10 查证）：conda env 里**没有** torch/torch_npu——它们在镜像系统
> python（/usr/local/python3.12.13）里，conda env 通过 site-packages 下的 `image.pth`
> 把该路径追加进 sys.path。conda env 自己的 transformers 5.12.1 在路径优先级上压过系统包。
> 所以：动 conda env 不影响 torch 栈；换镜像版本时要检查 `image.pth` 指向是否仍有效。

## 4. 安装 verl（含 3 个必带补丁）

补丁版已推送到 GitHub，直接 clone 即得（基线 `3c6d9b43` + padding.py jagged 修复 +
tensordict unbind 兜底 + tool_parser `...` 参数修复，commit `ef8e1216`）：

```bash
cd /workspace
git clone -b deepeyes-qwen35-npu https://github.com/archted/verl.git verl
cd /workspace/verl && /workspace/conda/verl/bin/pip install -e . --no-deps
```

配方（deepeyes 配置/奖励/工具）在 `/workspace/verl/recipe/deepeyes/`，随仓带出，无需另装。

## 5. 数据与模型

```bash
# 模型
huggingface-cli download Qwen/Qwen3.5-9B --local-dir /workspace/models/Qwen3.5-9B

# 数据：DeepEyes 47k 原始集 → 预过滤（必做，否则每次启动多等 13-16 分钟）
cd /workspace/verl && /workspace/conda/verl/bin/python recipe/deepeyes/scripts/prepare_data.py --output-dir /workspace/data/deepeyes
/workspace/conda/verl/bin/python recipe/deepeyes/scripts/prefilter_deepeyes.py
# 产出 train.filtered_8192.parquet（33137 条）/ val.filtered_8192.parquet（256 条）
```

数据红线：**Chart 子集必须剔除**（高分辨率多子图，训练必 OOM）；thinklite 子集分辨率过低
（裁剪 <1 个 patch，工具失效），预过滤脚本已处理。

## 6. judge 服务（另需一台 8 卡空闲机器）

```bash
# 在 judge 机容器内（142 实测配置），模型 Qwen3-30B-A3B-Instruct-2507：
/workspace/conda/verl/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name judge --tensor-parallel-size 8 \
  --max-model-len 8192 --port 18901 \
  --override-generation-config '{"max_new_tokens": 64}'
# 训练侧只需 export LLM_AS_A_JUDGE_BASE=http://<judge机IP>:18901/v1
```

## 7. 之后的步骤

训练与推理命令与一体镜像完全一致（环境等价），直接照 `ARCHTED_CASE.md` §5–§6 执行。
原始 138 容器与一体镜像的布局差异：138 容器的 `/workspace` 是宿主机 bind mount
（环境在宿主机上，容器 fs 里没有），一体镜像相反（环境烤在镜像里）——两者的
红线不同，见 CASE §7。
