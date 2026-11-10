# DeepEyes GRPO 案例复现手册（Qwen3.5-9B × Ascend 910B3）

> 维护人：archted ｜ 更新：2026-11-10 ｜ 本文所有版本号均为实测（非文档转抄）
> **本文主线 = 用一体镜像 `deepeyes-train-infer:v1` 复现训练与推理**（§3–§7）。
> 从零搭建新容器的旧路线已存档至 `ARCHTED_CASE_newbuild.md`（备查，优先走一体镜像）。

---

## 1. 案例是什么

用 verl 框架的 GRPO 算法，在昇腾 NPU 上端到端训练 Qwen3.5-9B 学会"边看图边思考"：
模型自主输出 bbox 坐标调用图像放大工具 → 裁剪图回灌 → 继续推理 → 给出答案。
奖励 = 0.8×答对(judge 判分) + 0.2×格式 + 1.2×真实用对工具。

**复现成功的验收标准**（本机实测参考值）：
- 训练能稳定跑完 step（无 OOM / aicore error）；
- `critic/score/mean` 从 ~1.1 上升至 1.26–1.38 区间（前 170 步实测）；
- 每 80 步 val：`val-core/vstar/acc/mean@1` 参考值 0.70–0.78；
- 138 单步耗时 ~29 分钟（TRAIN_BATCH_SIZE=64, ROLLOUT_N=8, 16384 响应长度）。

---

## 2. 镜像与环境版本（实测清单，2026-11-09 采集）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| 一体镜像 | **deepeyes-train-infer:v1（23.1GB）** | 见 §9，基线 `quay.io/ascend/vllm-ascend:v0.27.1rc1` |
| NPU | 8 × Ascend 910B3（HBM 64GB/卡） | 训练机 |
| NPU 驱动/HDK（npu-smi） | **25.5.2** | 138 实测；25.2.0 可跑推理但缺 ACL graph/IPC 加速 |
| 宿主机 OS | openEuler 22.03 LTS-SP2（kernel 5.10.0，aarch64） | |
| CANN Toolkit | **9.1.0**（V100R001C11SPC001B243） | 镜像内 `/usr/local/Ascend/ascend-toolkit/latest` |
| vLLM | **0.27.1+empty**（源码位于 /vllm-workspace/vllm） | 镜像内置 |
| vllm-ascend | **0.27.1rc1**（源码位于 /vllm-workspace/vllm-ascend） | 镜像内置 |
| torch / torch_npu | 2.10.0+cpu / **2.10.0.post4** | 镜像内置 |
| Python | 3.12.16（conda env `/workspace/conda/verl`） | |
| transformers | **5.12.1** | conda env 内（镜像系统 python 的 5.14.1 被覆盖，不生效） |
| verl | **0.10.0.dev0**（editable `/workspace/verl`） | 基线 commit `3c6d9b43` + 3 个 NPU 补丁 |
| 其他关键包 | datasets 5.1.0 / ray 2.48.0 / tensordict 0.10.0 / qwen-vl-utils 0.0.14 / accelerate 1.15.0 | |
| 模型 | Qwen/Qwen3.5-9B（HF，**不在镜像内**） | |
| judge 模型 | Qwen3-30B-A3B-Instruct-2507（TP=8，独立部署，**不在镜像内**） | 阅卷用，见 §6.1 |

> 一句话记忆：**镜像自带全套软件栈（训练+推理通用）；权重和数据不在镜像里，按 §4 准备。**

---

## 3. 导入镜像并起容器

```bash
# ① 导入（产物在 138：/home/h00516332/image_export/deepeyes-train-infer_v1.tar，22G）
docker load -i deepeyes-train-infer_v1.tar

# ② 起容器（驱动路径按新平台实际内核适配包路径调整）
docker run -d --name deepeyes --net=host --shm-size=64g --ulimit memlock=-1:-1 \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  deepeyes-train-infer:v1

# ③ 环境自检（30 秒）
docker exec deepeyes bash -c "/workspace/conda/verl/bin/python -c \
  'import torch, torch_npu, vllm, transformers, verl; print(transformers.__version__)' && \
  npu-smi info | head -4"
```

要点：`--net=host`（judge 跨机调用 + HCCL 端口）；9 个 davinci 设备一个不能少；
宿主机驱动目录 bind mount 进来（镜像内 CANN 与宿主机驱动版本要兼容）。

---

## 4. 准备数据与模型（镜像不含，二选一）

**方式 A：从 138 直接拷贝**（推荐，跳过下载和过滤）：

| 新平台目标路径 | 从 138 拷贝的源（宿主机 = 容器内 /workspace） | 大小 |
| --- | --- | --- |
| `/workspace/data/deepeyes/*.filtered_8192.parquet` | `/home/h00516332/deepeyes_run/data/deepeyes/` 里两个 filtered parquet | ~5G（可只拷 train+val 两个 filtered 文件） |
| `/workspace/models/Qwen3.5-9B` | `/home/h00516332/deepeyes_run/models/Qwen3.5-9B` | 18G |

**方式 B：重新下载生成**：

```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir /workspace/models/Qwen3.5-9B
cd /workspace/verl && /workspace/conda/verl/bin/python recipe/deepeyes/scripts/prepare_data.py --output-dir /workspace/data/deepeyes
/workspace/conda/verl/bin/python recipe/deepeyes/scripts/prefilter_deepeyes.py
# 产出 train.filtered_8192.parquet（33137 条）/ val.filtered_8192.parquet（256 条）
```

数据红线：**Chart 子集必须剔除**（高分辨率多子图，训练必 OOM）；thinklite 子集分辨率过低
（裁剪 <1 个 patch，工具失效），预过滤脚本已处理。

---

## 5. 跑训练

**前提检查**（全部满足才动手）：

```bash
npu-smi info                     # 8 卡全部空闲、Health OK
curl -s http://71.10.29.142:18901/v1/models   # judge 存活（换 judge 机则换 IP）
```

**启动**（容器内，或宿主机 docker exec）：

```bash
docker exec -d deepeyes bash -c "cd /workspace/verl && \
  HCCL_SOCKET_IFNAME=enp67s0f0np0 bash /workspace/run_full_138.sh \
  > /workspace/logs/full_outer_138_v3.log 2>&1"
```

`run_full_138.sh` 内含的关键开关（换机复现时逐一核对，缺一个都起不来或跑错）：

| 开关 | 值 | 为什么 |
| --- | --- | --- |
| `HCCL_SOCKET_IFNAME` | `enp67s0f0np0` | **网卡名因机器而异**，换机先 `ip addr` 查真实网卡名（脚本内 2 处 + 启动命令 1 处） |
| HCCL 端口范围 | 60000-60050 / 61000-61050 | 避开 EI0020 冲突 |
| `bypass_mode=True` | rollout 纠偏旁路 | 大幅提速（29min/步 vs 更慢），副作用：entropy 指标不再产出 |
| `weight_nz_mode=0` | 关闭 NZ 权重格式 | vllm-ascend 该版本要求 |
| `data.filter_overlong_prompts=False` | 关 | 已用预过滤数据替代（省 13-16 分钟/次启动） |
| `data.tool_config_path` | **绝对路径** | RLHFDataset 不解析相对路径（已知问题） |
| `max_actor_ckpt_to_keep=2` | 只留最近 2 个权重 ckpt | 每 ckpt 106GB，防磁盘爆 |
| `GEN_TP=2` | rollout 推理 TP=2 | 9B 在 2 卡 TP 下吞吐最优 |

**冒烟测试**（强烈建议先跑）：复制 `run_full_138.sh`，把
`data.train_batch_size=4`、`trainer.total_training_steps=1`、`MAX_RESPONSE_LENGTH=4096`
临时调小，10 分钟能验证全链路（rollout→工具→judge→反传→ckpt）。

**监控**：

```bash
tail -f /workspace/logs/full_outer_138_v3.log
grep -oE "critic/score/mean:[0-9.]+" <日志> | tail -5
grep -oE "val-core/[a-z_]+/acc/mean@1:[0-9.]+" <日志>  # 每80步的验证结果
```

---

## 6. 推理复现（评测）

推理有两层：**judge 阅卷服务**（另需一台 8 卡空闲机器）+ **被训模型的推理评测**。

### 6.1 judge 部署

```bash
# 在 judge 机容器内（142 实测配置）：
/workspace/conda/verl/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name judge --tensor-parallel-size 8 \
  --max-model-len 8192 --port 18901 \
  --override-generation-config '{"max_new_tokens": 64}'
# 训练侧只需 export LLM_AS_A_JUDGE_BASE=http://<judge机IP>:18901/v1
```

### 6.2 导出 RL checkpoint 为 HF 格式（纯 CPU，随时可做）

verl checkpoint 是 FSDP 8 分片（`ckpts/full/global_step_N/actor/model_world_size_8_rank_*.pt`），
推理前必须合并导出（**只有最新 2 个 step 含完整权重**）：

```bash
docker exec deepeyes bash -c "cd /workspace/verl && /workspace/conda/verl/bin/python \
  -m verl.model_merger merge --backend fsdp \
  --local_dir /workspace/ckpts/full/global_step_170/actor \
  --target_dir /workspace/models/eval_qwen35_9b_step170"
# 产出 ~18GB 单文件 safetensors + tokenizer；末尾 glibc abort 报错是已知无害噪声
```

### 6.3 起推理服务 + 跑评测

```bash
# ① 起服务（占 2 卡，TP=2；训练期间严禁，等 8 卡空闲）
docker exec -d deepeyes bash -c "cd /workspace && \
  ASCEND_RT_VISIBLE_DEVICES=0,1 bash /workspace/eval_prep/serve_eval_model.sh \
  /workspace/models/eval_qwen35_9b_step170 eval-qwen35-9b-step170 2 18900"

# ② 多轮 Agent 评测（先 --limit 16 冒烟，再全量 256 条 val）
docker exec deepeyes bash -c "export LLM_AS_A_JUDGE_BASE=http://71.10.29.142:18901/v1 && \
  /workspace/conda/verl/bin/python /workspace/eval_prep/eval_agent.py \
  --api-base http://127.0.0.1:18900/v1 --model eval-qwen35-9b-step170 \
  --data /workspace/data/deepeyes/val.filtered_8192.parquet \
  --out /workspace/eval_prep/results_step170 --concurrency 8"
```

脚本原理（`/workspace/eval_prep/eval_agent.py`，无需改动即可复现）：
模拟训练侧 AgentLoop —— 发图 → 解析 Qwen3.5 XML tool call → 0-1000 相对坐标裁图回灌
（最多 5 轮）→ 提取 `<answer>` → 发 judge 判分。指标与训练 reward 完全对齐：
`accuracy / tool_call_rate / format_rate / mean_score`，输出 `results.jsonl`（逐条）+ `summary.json`（汇总）。

### 6.4 纯推理冒烟（不起训练栈，30 秒验证环境）

```bash
docker exec deepeyes bash -c "/workspace/conda/verl/bin/python -c \"
import requests, json
r = requests.post('http://127.0.0.1:18900/v1/chat/completions', json={
  'model': 'eval-qwen35-9b-step170',
  'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}],
  'max_tokens': 16}, timeout=60)
print(r.json()['choices'][0]['message']['content'])\""
```

---

## 7. ⚠️ 红线（违反必翻车）

1. **严禁把宿主机目录整体挂载到 `/workspace`**——conda env 和 verl 源码烤在镜像的
   `/workspace/conda`、`/workspace/verl` 里，整体挂载会全部遮掉。需要持久化的数据按
   子目录挂载：`-v /xxx/data:/workspace/data`、`-v /xxx/logs:/workspace/logs`。
2. **`HCCL_SOCKET_IFNAME=enp67s0f0np0` 是 138 的网卡名**，换平台必须 `ip addr` 查真实网卡名后替换。
3. **驱动兼容**：镜像 CANN 9.1.0 要求宿主机 HDK ≥ 25.x（138 实测 25.5.2 通过；25.2.0 可推理，训练特性 ACL graph/IPC 缺失）。
4. judge 地址：`LLM_AS_A_JUDGE_BASE` 指向的 judge 服务要另行部署（§6.1），镜像不自带 judge 模型。

## 8. 坑速查（复现时必读）

| 坑 | 症状 | 解法 |
| --- | --- | --- |
| 网卡名不对 | HCCL 报错起不来 | `ip addr` 查真实网卡，改 `HCCL_SOCKET_IFNAME` |
| Chart 子集 | 训练 OOM | 预过滤剔除 |
| 相对 tool_config_path | 工具加载失败 | 配置里写绝对路径 |
| 3D jagged position_ids | 训练初期崩 | 镜像内 verl 已带补丁；若自行换 verl 源码必须用含 3 补丁的分支 |
| `...` 工具参数 | JSON 序列化崩溃 | 同上（tool_parser 补丁） |
| judge 400 | reward 异常记 0 | judge prompt 超长所致（已知问题，修复进行中） |
| checkpoint 磁盘 | 每 5 步 +106GB | `max_actor_ckpt_to_keep=2` 只留 2 个 |

## 9. 镜像导出存档（怎么造出来的，供重新构建）

### 9.1 镜像构成

`deepeyes-train-infer:v1`（23.1GB）= 基础镜像 `quay.io/ascend/vllm-ascend:v0.27.1rc1`
（提供 CANN 9.1.0 / torch 2.10.0.post4 / torch_npu / vllm 0.27.1 栈）
+ conda env（`/workspace/conda/verl`，480M）+ verl 源码（`/workspace/verl`，25M）
+ editable 依赖源（`/workspace/tmp/speculators-ascend-preflight`，14M）
+ 训练/评测脚本（`/workspace/run_full_138.sh`、`/workspace/eval_prep/`）。

**训练与推理同一镜像**：训练 rollout 与推理评测用同一 vLLM 栈，已在 138 实证。
**不含**（刻意排除，~390G）：模型权重、checkpoint、训练数据、ray_tmp、日志、各类缓存。

产物（138 宿主机）：`/home/h00516332/image_export/deepeyes-train-infer_v1.tar`（22G）
构建上下文（含 Dockerfile，可复现构建）：`/home/h00516332/deepeyes_run/image_build/`。

### 9.2 导出过程（2026-11-10 实测，供重新构建）

```bash
# ① staging：容器内 root 拷出必要物（bind mount 内容不在容器 fs 里，不能 docker commit！）
docker exec deepeyes bash -c "mkdir -p /workspace/image_build && cd /workspace && \
  tar -cf - --exclude='__pycache__' --exclude='*.pyc' \
    conda/verl verl tmp/speculators-ascend-preflight run_full_138.sh \
    eval_prep/eval_agent.py eval_prep/serve_eval_model.sh \
    | tar -C /workspace/image_build -xf - && \
  rm -rf /workspace/image_build/verl/.ruff_cache && chmod -R a+rX /workspace/image_build"
# ② Dockerfile（见 image_build/Dockerfile）：FROM v0.27.1rc1 + 5 条 COPY
cd /home/h00516332/deepeyes_run/image_build && docker build -t deepeyes-train-infer:v1 .
# ③ 验证（无 NPU 也能跑 import 测试）+ 导出
docker run --rm deepeyes-train-infer:v1 /workspace/conda/verl/bin/python -c \
  "import torch, torch_npu, vllm, transformers, verl; print(transformers.__version__)"
docker save -o /home/h00516332/image_export/deepeyes-train-infer_v1.tar deepeyes-train-infer:v1
```

> 为什么不用 `docker commit`：原容器的 conda env / verl / 脚本全在宿主机 bind mount
> `/workspace` 里（`docker diff` 证实），commit 会导出空壳，还会带走可写层 11k 个 pytest
> pyc 垃圾文件。Dockerfile 从干净基线重放，行为一致且可复现。
