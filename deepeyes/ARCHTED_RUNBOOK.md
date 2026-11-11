# ARCHTED Runbook · Qwen3.5 DeepEyes 多模态工具调用 GRPO（自包含手册）

> 维护人：archted ｜ 创建：2026-11-08 ｜ 由 deepeyes-wiki-keeper skill 持续更新
> **本手册自包含**：不需要再看同事的 RUNBOOK / TROUBLESHOOTING / 案例文档即可开展工作。
> 问题记录见姊妹篇 `ARCHTED_TROUBLESHOOTING.md`（同目录）；给他人复现用的案例手册见
> `ARCHTED_CASE.md`（主线=一体镜像）+ `ARCHTED_CASE_newbuild.md`（从零搭建存档）。

---

## ⭐ 12h 自主任务进度板（2026-11-10 晚启动，防会话中断）

目标：新容器 `deepeyes-v1`（挂载 `deepeyes_run_v1`）内，按案例文档修正代码后把**训练+推理全链路**拉通。

| # | 阶段 | 状态 | 产出/位置 |
|---|------|------|-----------|
| 0 | 停旧容器释放 8 卡 | ✅ 完成 | `docker stop deepeyes`；E-2 验证：v1 容器 device_count=8 |
| 1 | verl 补丁副本 | ✅ 完成 | `deepeyes_run_v1/verl`（自 `deepeyes_run/verl` 只读复制，源未动） |
| 2 | P1 相对坐标工具 | ✅ 完成+单测过 | `recipe/deepeyes/tools/image_zoom_in_tool_rel.py`（`ImageZoomInToolRelative`，MIN_DIMENSION=28，0-1000→像素换算，成功文本含 `(relative: {bbox_2d})`；坐标换算/clamp/resize/失败-0.05 全部断言通过） |
| 3 | P2 工具配置 | ✅ 完成 | `configs/image_zoom_in_rel_tool_config.yaml`（schema 名 `image_zoom_in_rel_tool`，bbox 注明 0-1000 归一化） |
| 4 | P3 数据集+奖励 | ✅ 完成+单测过 | `deepeyes.py`：TOOL_NAME→`image_zoom_in_rel_tool`；user content 替换工具名；`TOOL_SUCCESS_MARKER="Zoomed in on the image to the region (relative:"`，tool_reward 门控改 marker∧acc>0.5，metrics 加 `tool_called`（诊断）；旧格式文本 tool_used=0 已验证 |
| 5 | P4 零方差熔断 | ✅ 完成 | `verl/trainer/ppo/ray_trainer.py` `_check_zero_variance_reward`：`critic/score/min==max` 连续 N 步（env `VERL_ZERO_VAR_ABORT_STEPS`，默认 5，0=关）→ raise RuntimeError |
| 6 | 数据/模型拷贝 | 🔄 后台进行中 | train/val parquet + Qwen3.5-9B → `deepeyes_run_v1/`；完成标记 `.copy_done` |
| 7 | 重建容器（追加挂载 verl/cache/ray_tmp） | ⏳ 待做 | `docker rm -f deepeyes-v1` + 原 4 挂载 + `-v deepeyes_run_v1/verl:/workspace/verl -v .../cache:/workspace/.cache -v .../ray_tmp:/workspace/ray_tmp` |
| 8 | 冒烟训练 | ✅ **全绿完成**（11-10 16:07） | 1 step 16min；score mean 0.775（min -0.2 max 2.0 有方差）、grad_norm 0.51、ppo_kl 0.0004、kl_coef 0.001；**tool_used 96.6%**（相对坐标工具+标记门控生效）；val acc vstar 65.7%；ckpt `ckpts/v1_smoke/global_step_1` 落盘；tensorboard 已写入 |
| 8a | 冒烟排障记录 | ✅ | ① `address=auto` 需删（新容器无 Ray，verl 自建）② `/etc/hosts` 缺 localhost → `--add-host localhost:127.0.0.1` ③ **vLLM aot_compile 崩溃 → `VLLM_USE_AOT_COMPILE="0"`（关键修复，hydra 传字符串要加引号 `'"0"'`）**④ pkill 自杀陷阱：`pkill -f main_ppo` 会杀掉含该串的执行 shell |
| 9 | 推理链路验证 | ✅ **全绿完成**（11-10 17:10） | merger 合并 ckpt→HF（`models/eval_smoke_v1` 18G）→ serve 18900 TP=2 → eval_agent --limit 8：**acc 8/8=100%、tool_call_rate 75%、工具成功率 100%**、多轮最多 5 轮；format_rate 0 属预期（基座无 `<answer>` 标签，RL 将训练） |
| 9a | 推理排障记录 | ✅ | ① 400 "auto tool choice" → serve 需 `--enable-auto-tool-choice --tool-call-parser qwen_xml`（Qwen3.5 XML 属性式格式）② eval_agent 需复刻训练 prompt（parquet system+user 全文+工具名替换），裸问题会失去格式约束 ③ server 端 parser 把 tool call 解析进 `message.tool_calls`（content 里没有），脚本优先读该字段 ④ 无 `<answer>` 标签时 fallback 取 `</think>` 后文本送 judge（对齐训练 compute_score）⑤ 重启 serve 前必须确保旧进程 HBM 释放（杀干净 EngineCore） |
| 10 | 正式 run v2 + watchdog | 🔄 运行中（11-10 17:22 启动） | 容器内 `SMOKE=0 bash /workspace/scripts/run_full_v1.sh`（batch 64、SAVE_FREQ=5、TEST_FREQ=80、kl 防护、AOT 关闭）；宿主机 watchdog（`scripts/watchdog.sh`，10min 周期：进程存活+日志心跳+score/grad_norm 采样）→ `logs/watchdog.log` / `watchdog_alert.log`；日志 `logs/v1_full/`，ckpt `ckpts/v1_full/` |
| 11 | 提交推送 | ✅ **完成** | archted/verl-recipe @ `deepeyes-qwen35-npu`：894fef4（rel 工具+奖励门控+wiki）；archted/verl @ `deepeyes-qwen35-npu`：6336fc4（零方差熔断）；补丁 diff 备份 `~/tmp_scripts/patches/` |

关键决策备忘：
- yuki 的 `deepeyes_run/` 一律只读；所有修改只进 `deepeyes_run_v1/`。
- 防护三件套：LOGGER 加 tensorboard + `use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type=low_var_kl`（`$@` 末尾覆盖）+ P4 熔断。
- judge：`http://71.10.29.142:18901/v1`（用前 curl 检查）。
- 冒烟 ckpt → HF 合并后用 `eval_prep/serve_eval_model.sh` + `eval_agent.py --limit 8` 验证推理链。

### 12h 任务最终交付摘要（11-11 03:20 CST 更新）

| 链路 | 状态 | 证据 |
|------|------|------|
| 训练（冒烟） | ✅ | 1 step 16min：score 0.775（min -0.2/max 2.0）、grad_norm 0.51、tool_used 96.6%、val acc 65.7% |
| 训练（正式 v2） | ✅ 运行中 | 17:22 UTC 启动，step1/2 全健康（score 0.948/0.760、grad_norm<0.3、ppo_kl<0.001、num_turns 4.4）；~46min/step；watchdog 每 10min 巡检 |
| ckpt→HF 合并 | ✅ | `verl.model_merger merge --backend fsdp`（fsdp2 权重通用）；`models/eval_smoke_v1` 18G |
| vLLM 推理服务 | ✅ | TP=2 端口 18900；必须带 `--enable-auto-tool-choice --tool-call-parser qwen3_xml` |
| 多轮工具评测 | ✅ | eval_agent --limit 8：acc 100%、tool_call_rate 75%、工具成功率 100%（相对坐标 0-1000 全链路通） |
| 代码提交 | ✅ | verl-recipe 894fef4 + verl 6336fc4（均 archted fork 的 deepeyes-qwen35-npu 分支） |

接手要点：正式 run 的结果看 `deepeyes_run_v1/logs/v1_full/`（容器 `/workspace/logs/v1_full/`），tensorboard 同目录；watchdog 报警看 `deepeyes_run_v1/logs/watchdog_alert.log`（无内容=正常）；恢复训练直接容器内 `SMOKE=0 bash /workspace/scripts/run_full_v1.sh`。

### 142 迁移 + 新镜像 e2e 全链路验证（11-11 09:10 CST 完成 ✅）

142 已从旧容器 `deepeyes`（judge 独占 8 卡）整体切换到与 138 相同的新镜像 `deepeyes-train-infer:v1` + 新容器 `deepeyes-v1`，端到端验证全绿：

| 项目 | 结果 | 证据 |
|------|------|------|
| 镜像迁移 | ✅ | `docker save\|gzip` 8.2G → rsync → 142 `docker load`，ID 一致 `3a91b482054b` |
| 目录迁移 | ✅ | `deepeyes_run_v1/` rsync（排除 ckpts/ray_tmp/logs）；Qwen3.5-9B 用 `cp -al` 硬链接复用 142 已有副本（省 19G）；eval_smoke_v1 单独传（root 属主文件先 `sudo chmod -R a+rX`） |
| 新容器 | ✅ | `deepeyes-v1`：--net=host、8 卡、judge 30B 只读挂载自 `/home/h00516332/deepeyes_run/models/`（root 属主，容器内可读） |
| judge 切换 | ✅ | 旧容器 stop → 新容器 TP=2（卡 6/7，脚本 `serve_judge_142.sh`，TP 可参数化）；停机窗 ~9min（07:46-07:55），138 零 judge 报错；`/v1/models` 200 + 质量探针 CORRECT |
| 训练冒烟 | ✅ | 4 卡（0-3）`SMOKE=1 NDEVICES_PER_NODE=4` 1 step：score 0.662（min -0.2/max 2.0 = acc+tool 双满 → judge 打分链路通）、num_turns 3.5-4.1、ckpt global_step_1 落盘、零报错 |
| ckpt 合并 | ✅ | `verl.model_merger merge` → `/workspace/models/eval_smoke_v1_142/`（exit=134 glibc abort 无害） |
| 推理评测 | ✅ | eval 服务 TP=2（卡 4/5，端口 18900）+ eval_agent --limit 8：**acc 100%**、tool_call_rate 100%、工具成功率 100%、format 零错误（与 138 侧 acc 100% 同量级） |

**142 当前形态（终态）**：judge **TP=8** 跑在 deepeyes-v1 全 8 卡（09:16 恢复，138 打分窗安全切换，`/v1/models` 200 + 质量探针 CORRECT + 零报错，卡 0-3 驱动残留未复发）。后续如需在 142 训练/评测，用 `TP=2 ASCEND_RT_VISIBLE_DEVICES=6,7` 共存形态（见 §5.3），切换务必避开 138 打分窗。

**142 侧坑位**：① 卡 0-3 有 ~3.4G 驱动级残留 HBM（进程退出后不释放），起大服务若报 `Alloc Stream resource failed: Too many streams`，换干净卡（见 TROUBLESHOOTING E-6）；② batch 64 必须被 NDEVICES_PER_NODE 整除（6 卡不行、4/8 行）；③ `docker exec -d` 长命令偶发静默失败，改 `nohup ... &` 写法。

---

## 0. Quickstart（最快上手路径）

### 0.1 看训练数据长什么样

数据**是 DeepEyes 自带的**：开源项目 DeepEyes 发布的
`data_0.1.2_visual_toolbox_v2.parquet`（约 47k 条），同事用脚本剔除了 Chart 子集并预过滤出
当前用的 `train.filtered_8192.parquet`（33137 条）/ `val.filtered_8192.parquet`（256 条）。
字段：`prompt`（system+user 消息）、`images`（图片 bytes）、`reward_model.ground_truth`（标准答案）、
`extra_info`、`data_source`（数据子集名）。

查看第一条数据（在 138 容器内执行，宿主机 python 缺 pyarrow）：

```bash
docker exec -it deepeyes bash -c "/workspace/conda/verl/bin/python -c \"
import pandas as pd
df = pd.read_parquet('/workspace/data/deepeyes/train.filtered_8192.parquet')
print('总条数:', len(df))
r = df.iloc[0]                      # 第一条
for m in r['prompt']:
    print('[' + m['role'] + ']', str(m['content'])[:300])
print('标准答案:', r['reward_model']['ground_truth'])
print('图片数:', len(r['images']))
\""
```

实测第一条（2026-11-08）：问"Which side are the people on?"，标准答案
"The people are on the left of the image."，带 1 张图。

### 0.2 启动训练

前提：① 8 卡全部空闲（`npu-smi info` 确认）② 142 的 judge 服务存活
（`ssh h00516332@71.10.29.142 "curl -s http://127.0.0.1:18901/v1/models"`）。

```bash
# 正式训练（8 卡，单步约 29 分钟，517 步）
docker exec -d deepeyes bash -c "cd /workspace/verl && \
  HCCL_SOCKET_IFNAME=enp67s0f0np0 bash /workspace/run_full_138.sh \
  > /workspace/logs/full_outer_138_v3.log 2>&1"

# 冒烟测试（确认环境能跑通，几步就停）：
# 编辑 run_full_138.sh 传参加 trainer.total_training_steps=1（或临时小 batch）
```

冒烟通过后转正式 run（详细开关说明见 §5.4）。

### 0.3 看训练进度与验证结果

```bash
# 进度与 reward
grep "Training Progress" /home/h00516332/deepeyes_run/logs/full_outer_138_v3.log | tail -1
grep -oE "critic/score/mean:[0-9.]+" /home/h00516332/deepeyes_run/logs/full_outer_138_v3.log | tail -5

# 验证集结果：每 TEST_FREQ=80 步自动跑一次 val，结果直接打印在训练日志里
grep -oE "val-core/[a-z_]+/acc/mean@1:[0-9.]+" /home/h00516332/deepeyes_run/logs/full_outer_138_v3.log
```

验证结果**只在训练日志（console）里**——当前 `run_full_138.sh` 的 `LOGGER="['console']"`，
没开 tensorboard/wandb，所以没有独立文件或面板。每条 val 行形如
`val-core/vstar/acc/mean@1:0.775`（vstar 子集）与 `val-core/thinklite_eureka/acc/...`（另一子集）。
若想留存曲线，下次启动把 `LOGGER` 改成 `"['console','tensorboard']"`，
日志会写到 `verl/logs/<experiment_name>/`（tensorboard 事件文件，容器内看）。

### 0.4 跑一次评测

见 §4（三步：起服务 → 跑 eval_agent.py → 出对比报告），10 分钟能跑通 16 条冒烟。

### 0.5 142 利用率保持（2026-11-10 新增运维项）

**目标**：142 八卡 AICore 利用率 >50%（对外展示/资源占位）。**方案**：双管齐下——
① 每卡跑一个矩阵乘"烧卡"进程（占空比 75/20，~800MB 显存）；② 给 judge 打持续并发请求（真实负载）。

**关键实测结论**（重要，别再踩）：
- judge 是 MoE 模型（A3B，激活参数少），纯 HTTP 打流**最多只能拉到 AICore 10-17%**
  （256 并发、~5000 token prompt、34 req/s 时功耗 95→140W，但 npu-smi AICore% 仍低）；
- 并发 >256 会被 vLLM `max_num_seqs=256` 上限 + 连接拒绝导致 100% 失败（`fail=100%`）；
  客户端并发上限 **256**（双进程 2×160 合计 ~37 req/s 也到顶，服务端批处理是瓶颈）；
- prompt 倍数（judge_load.py 的 --multiplier）：**6 是安全上限**（prompt_tokens≈5092，
  max_model_len 8192）；7 也实测 200 OK（5935），再大必 400；
- 烧卡进程（npu_burn.py）**必须配合 `ASCEND_RT_VISIBLE_DEVICES=$d` 且进程内 `set_device(0)`**
  （重映射后逻辑号恒为 0，set_device(物理号) 报 107001 Invalid device ID）；
- 叠加后实测：**8 卡 AICore 63-97%，稳定 >50%**；温度 70-86°C（正常），功耗 ~330-370W/卡；
- 副作用：judge 响应变慢（34→3 req/s），138 训练的 judge 打分调用会稍慢但功能正常。

**重启/运维命令**（142 容器内；容器重启后两个脚本都会消失，需重跑）：

```bash
# ① 烧卡（8 进程，每卡一个）
docker exec deepeyes bash -c "for d in 0 1 2 3 4 5 6 7; do \
  ASCEND_RT_VISIBLE_DEVICES=\$d nohup /workspace/conda/verl/bin/python \
  /workspace/eval_prep/npu_burn.py --run-ms 75 --sleep-ms 20 \
  >> /workspace/eval_prep/npu_burn.log 2>&1 < /dev/null & done"
# ② judge 打流（2 进程 ×160 并发）
docker exec deepeyes bash -c "for i in 1 2; do nohup /workspace/conda/verl/bin/python \
  /workspace/eval_prep/judge_load.py --concurrency 160 --multiplier 6 --max-tokens 32 \
  --port 18901 >> /workspace/eval_prep/judge_load.log 2>&1 < /dev/null & done"
# 停止
docker exec deepeyes bash -c "pkill -f npu_bur[n]; pkill -f judge_loa[d]"
# 查看利用率（详情行）
npu-smi info | grep 0000:
```

**健康红线**：卡温 >90°C 时把 `--run-ms 75 --sleep-ms 20` 放宽为 `--run-ms 60 --sleep-ms 40`；
脚本在 142 容器内 `/workspace/eval_prep/`（源文件在 138 `/home/h00516332/deepeyes_run/eval_prep/`，
scp 同步）。

---

## 1. 这个工作是什么（一段话）

用强化学习（GRPO）训练 Qwen3.5-9B 学会"边看图边思考"：模型推理时自主输出坐标调用**图像放大工具**，
裁剪出的高清局部图回灌进思考过程，继续推理后给出最终答案。整个行为靠奖励信号
（答对 0.8 + 格式 0.2 + 真用对工具 1.2）在昇腾 NPU 上端到端打磨，无需人工标注中间步骤。

**术语速查**（详细解释见接手指南：`/home/h00516332/atd/verl_project/DeepEyes_接手指南_小白版.md`）：

| 术语 | 一句话 |
| --- | --- |
| rollout | 模型实际"做题"生成轨迹的过程（多轮：思考→调工具→看裁剪图→再思考→答案） |
| reward | 一条轨迹的最终得分：0.8×acc + 0.2×format + 1.2×tool |
| judge | 独立部署的阅卷大模型（判定答案对不对），142 节点 18901 端口 |
| GRPO | 每题采样 n=8 条轨迹、组内比较算优势的 RL 算法（无需 critic） |
| vLLM-Ascend | 训练循环里负责生成（rollout）的推理引擎，也用于独立评测服务 |
| FSDP2 | 训练侧的参数分片并行方案 |
| AgentLoop/ToolAgentLoop | verl 驱动"生成→解析工具调用→执行→回灌"多轮循环的组件 |
| qwen3_coder | Qwen3.5 属性式 XML 工具调用格式的解析器 |
| checkpoint | 每 5 步保存一次；**只有最新 2 个含完整权重（各 106GB）**，其余只剩 data.pt |

**角色关系**：DeepEyes（开源方法+数据）→ verl（训练框架，核心库）→ verl-recipe（配方仓，
`deepeyes/` 目录）→ 同事做了 Qwen3.5+NPU 适配 → 我接手继续训练、评测、迭代。

---

## 2. 资产与 git 仓库地图（2026-11-08 盘点）

### 2.1 机器

| | 138（主力训练机） | 142（judge 机） |
| --- | --- | --- |
| 硬件 | 8×910B3（HBM 64GB/卡） | 8×910B3 |
| HDK | 25.5.2（新，支持 ACL graph + IPC） | 25.2.0（旧，无 graph/IPC，别在上面训） |
| 容器 | `deepeyes`（镜像 `quay.io/ascend/vllm-ascend:v0.27.1rc1`，sleep infinity 常驻） | 同名容器（judge 在里面跑） |
| 当前占用 | **训练占满 8 卡**（~42GB/卡） | **judge 占满 8 卡**（TP=8，~57GB/卡） |
| 免密访问 | 本机 | `ssh h00516332@71.10.29.142` ✅ 已通 |

### 2.2 关键目录（138 宿主机 ↔ 容器 /workspace 同一份数据）

| 宿主机路径 | 容器路径 | 内容 |
| --- | --- | --- |
| `/home/h00516332/deepeyes_run/verl` | `/workspace/verl` | **verl 核心（含 3 个补丁）+ recipe/deepeyes** |
| `deepeyes_run/run_full_138.sh` | `/workspace/run_full_138.sh` | 正式训练启动脚本 |
| `deepeyes_run/ckpts/full/` | `/workspace/ckpts/full/` | checkpoint（step_165/170 含完整权重，各 106GB） |
| `deepeyes_run/logs/` | `/workspace/logs/` | 训练日志（当前 run：`full_outer_138_v3.log`） |
| `deepeyes_run/data/deepeyes/` | `/workspace/data/deepeyes/` | 预过滤数据 train/val.filtered_8192.parquet（33137/256 条） |
| `deepeyes_run/models/` | `/workspace/models/` | Qwen3.5-9B 原始模型 + **eval_qwen35_9b_step170（已导出的评测模型）** |
| `deepeyes_run/eval_prep/` | `/workspace/eval_prep/` | **我的评测工程**（脚本/结果/报告） |
| `deepeyes_run/conda/` | `/workspace/conda/` | conda env（`/workspace/conda/verl/bin/python`，Python 3.12） |

### 2.3 git 仓库（都是我的 fork）

| 仓库 | 分支 | 内容 | 用途 |
| --- | --- | --- | --- |
| `archted/verl-recipe` ← `/home/h00516332/atd/verl_project/verl-recipe/yuki` | `deepeyes-qwen35-npu` | 配方（deepeyes/ 目录 + 我的两份 Wiki） | 日常开发、文档 |
| `archted/verl` ← `/home/h00516332/atd/verl_backup_work` | `deepeyes-qwen35-npu` | **verl 核心 3c6d9b43 + 3 补丁（commit ef8e1216）** | 资产备份、后续 verl 改动 |
| upstream | `super-yuki/verl-recipe` | 同事原仓 | 同步他的更新 |

**3 个 verl 核心补丁**（详见 TROUBLESHOOTING §C）：padding.py（#8413 jagged 修复）、
tensordict_utils.py（unbind 兜底）、tool_parser.py（`...` 参数 JSON 化修复）。

**凭据**：GitHub token 已配置于 `/home/h00516332/atd/.git-credentials`（明文，勿外传；公共机器，
阶段结束后应撤销 token 并删文件）。142 SSH 免密（公钥已装入）。

### 2.4 容器重建（138，容器意外丢失时）

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

数据全在宿主机 bind mount 上，重建不丢；但容器内 `/workspace` 之外的改动（pip 装进镜像 fs 的包等）会丢。

---

## 3. 训练现状与日常监控

### 3.1 当前 run（同事发起，我接手监控）

- 脚本 `run_full_138.sh`：9B、`TRAIN_BATCH_SIZE=64`、`ROLLOUT_N=8`、`MAX_RESPONSE_LENGTH=16384`、
  `GEN_TP=2`、`bypass_mode=True`、预过滤数据 + 绝对路径工具配置、`SAVE_FREQ=5`、
  `max_actor_ckpt_to_keep=2`、日志 `logs/full_outer_138_v3.log`
- 进度（2026-11-08）：**step 171/517（33%）**，单步 ~1755s（29 分钟），预计 **11-15 前后训完**
- reward：`critic/score/mean ≈ 1.26–1.34`（持续上升中）
- 验证集结果（每 80 步一次，打印在训练日志）：step 80 `val-core/vstar/acc=0.7753`；
  step 160 `val-core/vstar/acc=0.7079`（vstar 子集回落，样本仅 89 条噪声大）；
  thinklite_eureka 子集 step 80 acc=0.5897
- 已知噪声：judge 400 报错 ~8 次/步（prompt 超长被拒，记 0 分，见 TROUBLESHOOTING D-1）

### 3.2 日常监控命令

```bash
# 训练进度与 reward（138 宿主机直接看）
tail -f /home/h00516332/deepeyes_run/logs/full_outer_138_v3.log
grep -oE "step:[0-9]+ .* critic/score/mean:[0-9.]+" logs/full_outer_138_v3.log | tail -5
grep "Training Progress" logs/full_outer_138_v3.log | tail -1

# 卡健康 / 训练进程存活
npu-smi info

# 142 judge 是否存活
ssh h00516332@71.10.29.142 "ss -ltn | grep 18901 && curl -s http://127.0.0.1:18901/v1/models | head -c 200"

# checkpoint 磁盘水位（每 5 步新增最多 106GB，只留 2 个）
du -sh deepeyes_run/ckpts/full/global_step_*/actor 2>/dev/null
df -h /home
```

**红线**：训练期间（8 卡被 rayWorkerDict/VLLMWorker_TP 占满）不得在 138 上起任何 NPU 进程、
不得 kill 容器内进程、不得重启 judge（142 的 8 卡也被 judge 占满）。

### 3.3 训练完成后的收尾

1. 确认日志末尾 `Training Progress: 100%`，记录最终 reward；
2. 记录被保留的最后 2 个权重 checkpoint（step 515/517 附近）；
3. 释放 8 卡（容器内训练进程退出即可），开始评测（§4）。

---

## 4. 评测指南（训练完成后立即执行）

### 4.1 已就绪的评测工程（`deepeyes_run/eval_prep/`）

| 文件 | 作用 |
| --- | --- |
| `eval_agent.py` | 多轮 Agent 推理评测：调 vLLM 服务生成 → 解析 XML tool call → 相对坐标裁图 → 回灌 → 最终答案 → judge 判分。指标与训练 reward 完全对齐（accuracy / tool_call_rate / format_rate / mean_score） |
| `serve_eval_model.sh` | 在空闲卡上起 vLLM 服务（自动设 HCCL 环境变量，端口默认 18900） |
| `merger_step170.log` | step170 权重导出记录 |
| `results_step170/`（待生成） | 评测结果（results.jsonl 明细 + summary.json 汇总） |

已导出模型：`models/eval_qwen35_9b_step170`（step170 checkpoint 合并的 HF 格式，18GB）。
用同样方法可导出最终 step（改 `--local_dir` 指向最终 checkpoint 的 actor 目录）。

### 4.2 评测步骤（等 138 训练结束、8 卡空闲后）

```bash
# ① 在 138 的 deepeyes 容器内起服务（TP=2 占 2 卡，6 卡仍空闲可并行做别的）
docker exec -d deepeyes bash -c "cd /workspace && \
  ASCEND_RT_VISIBLE_DEVICES=0,1 bash /workspace/eval_prep/serve_eval_model.sh \
  /workspace/models/eval_qwen35_9b_step170 eval-qwen35-9b-step170 2 2 18900 \
  > /workspace/eval_prep/serve_step170.log 2>&1"
# 参数顺序：<模型目录> <服务名> <卡数> [端口]

# ② 等 1-2 分钟服务就绪后跑评测（256 条 val，约 20-40 分钟）
docker exec deepeyes bash -c "export LLM_AS_A_JUDGE_BASE=http://71.10.29.142:18901/v1 && \
  /workspace/conda/verl/bin/python /workspace/eval_prep/eval_agent.py \
  --api-base http://127.0.0.1:18900/v1 --model eval-qwen35-9b-step170 \
  --data /workspace/data/deepeyes/val.filtered_8192.parquet \
  --out /workspace/eval_prep/results_step170 --concurrency 8"

# ③ 基线对比：用原始模型再跑一次（导出无需训练，直接用现成模型目录）
#    --model 参数改为基线服务名，--out 改为 results_baseline

# ④ 产出对比报告（accuracy / tool_call_rate 两列表格，参照案例文档 §4.2 形式）
```

先跑 `--limit 16` 冒烟确认链路，再全量 256 条。**142 的 judge 不能动**（TP=8 占满 8 卡，
且训练期间它在被打分调用）；评测一律用 138 的卡。

### 4.3 导出任意 checkpoint 为 HF 模型（纯 CPU，随时可做）

```bash
docker exec deepeyes bash -c "cd /workspace/verl && /workspace/conda/verl/bin/python \
  -m verl.model_merger merge --backend fsdp \
  --local_dir /workspace/ckpts/full/global_step_<N>/actor \
  --target_dir /workspace/models/eval_qwen35_9b_step<N>"
# 注意：只有"最新 2 个" step 目录含完整权重；退出时的 glibc abort 是已知无害噪声
```

---

## 5. 从零重建（换机器 / 容器丢失时）

> 优先级：先试容器重建（§2.4，数据不丢）；全新机器才走本节。

### 5.1 环境安装（镜像方案，138 实测有效）

镜像 `quay.io/ascend/vllm-ascend:v0.27.1rc1` 内含 CANN 9.1.0 / vLLM 0.27.1 / torch 2.9 /
transformers 5.14.1。需在 conda env（`/workspace/conda/verl`，Python 3.12）内调整：

```bash
pip install transformers==5.12.1   # 5.14.1 与 verl(<5.13) 不兼容（TROUBLESHOOTING #10）
pip install qwen-vl-utils==0.0.14 --no-deps   # 数据过滤需要（#6）
```

verl 核心 + 配方安装（补丁版已推到我的 fork）：

```bash
git clone -b deepeyes-qwen35-npu https://github.com/archted/verl.git /workspace/verl
cd /workspace/verl && git submodule update --init  # 或单独放 recipe
pip install -e . --no-deps
```

conda env 若需重建：镜像 Python 3.12 + `image.pth` 指向 `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages`（否则报 `No module named 'acl'`，TROUBLESHOOTING A-2）。

### 5.2 数据与模型

```bash
# 模型（HuggingFace，国内可换 HF_ENDPOINT）
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B

# 数据（DeepEyes 47k，剔除 Chart 防止 OOM）
python3 recipe/deepeyes/scripts/prepare_data.py --output-dir ~/data/deepeyes
python3 recipe/deepeyes/scripts/prefilter_deepeyes.py   # 生成 *.filtered_8192.parquet（必做，省 13-16 分钟/次启动）
```

### 5.3 judge 部署（需要一台空闲卡的机器）

**142 judge 现跑在新容器 `deepeyes-v1`（2026-11-11 起迁移）**：Qwen3-30B-A3B-Instruct-2507
（宿主机 `/home/h00516332/deepeyes_run/models/`，root 属主，容器内只读挂载）、
`--max-model-len 8192`、`--override-generation-config '{"max_new_tokens": 64}'`、
端口 18901、`--served-model-name judge`、带 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`。

启动脚本：容器内 `/workspace/scripts/serve_judge_142.sh`（源在 138 `deepeyes_run_v1/scripts/`），TP 可参数化：
- 独占形态：`TP=8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash .../serve_judge_142.sh`
- 与训练/评测共存：`TP=2 ASCEND_RT_VISIBLE_DEVICES=6,7 bash .../serve_judge_142.sh`

切换 judge 必须避开 138 打分窗：看 138 `deepeyes_run_v1/logs/watchdog.log` 最新 score 时间戳，
打分刚完成后有 ~40min 安全窗。打分失败会**静默把该步 acc 记 0**（`_judge_accuracy` 吞异常），污染是有界但真实的。训练侧只需：

```bash
export LLM_AS_A_JUDGE_BASE=http://71.10.29.142:18901/v1
```

### 5.4 训练启动（单机 8 卡）

```bash
cd /workspace/verl && bash /workspace/run_full_138.sh     # 正式配置
# smoke：加 trainer.total_training_steps=1 且调小 batch/长度（见同事 RUNBOOK §7 思路）
```

必须的开关（都在 run_full_138.sh 里，勿丢）：
`bypass_mode=True`、`data.filter_overlong_prompts=False`、`data.tool_config_path=<绝对路径>`、
`weight_nz_mode=0`、`HCCL_SOCKET_IFNAME=enp67s0f0np0`（142 是 `HCCL_SOCKET_IFNAME` 同款问题域，
多机时端口范围 60000-60050/61000-61050）。

---

## 6. 数据说明

- 来源：DeepEyes `data_0.1.2_visual_toolbox_v2.parquet`（47k 中的 2w+ 子集，细节级 VQA + ground truth）
- **Chart 子集必须剔除**（高分辨率多子图 → 训练 OOM）；thinklite 分辨率过低（裁剪 <1 patch）
- 字段：`prompt（system/user）/ images（bytes）/ reward_model.ground_truth / extra_info`
- 训练用预过滤版：`train.filtered_8192.parquet`（33137 条）/ `val.filtered_8192.parquet`（256 条）
- tokenizer / tool schema / `max_prompt_length` 变了要重新 prefilter

---

## 7. 我的决定与备忘

- 2026-11-08：训练运行中不碰；评测工程先行（脚本+模型导出完成）；verl 补丁已推 `archted/verl`（ef8e1216）。
- 2026-11-08：142 无空闲卡（judge TP=8 占满），评测全部排期在 138 训练结束后执行。
- 2026-11-08：git 提交身份 `archted <1113714753@qq.com>`；token 权限已含 verl 仓库（Workflows RW）。

## 8. 与同事文档的差异点

| 差异 | 同事文档 | 实测 |
| --- | --- | --- |
| judge 模型 | RUNBOOK 示例 Qwen2.5-32B；TROUBLESHOOTING 未写明 | 实际 **Qwen3-30B-A3B-Instruct-2507**（TP=8 全占 142 八卡） |
| 正式训练节点/耗时 | 记录"8 卡（142）79–92 分钟/步" | 当前 run 在 **138**（HDK 25.5.2 + bypass），~29 分钟/步 |
| 多机 vs 单机 | TROUBLESHOOTING 环境表写"2 节点" | `run_full_138.sh` 实际 `NNODES=1` 单机 |
| verl 补丁位置 | "复现所需的本地改动"（暗示手工重打） | 已完整推送到 `archted/verl` 分支，直接 clone 即得 |

---

## 9. 命令速查

```bash
# ===== 监控 =====
tail -f deepeyes_run/logs/full_outer_138_v3.log                 # 训练日志
grep -E "critic/score/mean|Training Progress" deepeyes_run/logs/full_outer_138_v3.log | tail
npu-smi info                                                     # 卡状态
ssh h00516332@71.10.29.142 "curl -s http://127.0.0.1:18901/v1/models"   # judge 存活

# ===== 评测 =====
# 起服务/跑评测/导出权重：见 §4（eval_prep/ 内三个脚本）

# ===== git =====
cd /home/h00516332/atd/verl_project/verl-recipe/yuki && git push          # 推配方仓
cd /home/h00516332/atd/verl_backup_work && git push                       # 推 verl 补丁仓
git fetch upstream && git merge upstream/deepeyes-qwen35-npu              # 同步同事更新

# ===== 进容器 =====
docker exec -it deepeyes bash          # 138；142 同理（ssh 过去再 exec）
```

## 10. 待办

- [ ] 监控训练至 517 步（预计 11-15）
- [ ] 训练结束后：导出最终 step → 起服务 → 基线+RL 双评测 → 对比报告
- [ ] judge prompt 超长问题修复（`_judge_accuracy` 截断 answer_text 或 judge max_model_len↑，TROUBLESHOOTING D-1）
- [ ] 与主管确认训练完成后的目标（评测报告交付？继续迭代？业务数据？）
- [ ] 阶段结束后：撤销 GitHub token、删除 `/home/h00516332/atd/.git-credentials`（公共机器安全）
