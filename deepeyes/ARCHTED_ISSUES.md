# ARCHTED Issues · 问题挖掘与提交台账

> 维护人：archted ｜ 创建：2026-11-08 ｜ KPI 驱动：发现并解决问题（DTS 单 + 开源 issue/PR）
> 由 deepeyes-wiki-keeper skill 持续更新。**铁律：提交前先对照 §2 去重，先签 CLA 再提 PR。**

---

## 1. 渠道与选择标准

| 渠道 | 适合什么 | 注意 |
| --- | --- | --- |
| **开源 issue/PR（verl-project/verl）** | verl 核心与 recipe/deepeyes 的代码 bug、改进；能给出复现/证据 | **PR 前必须签 CLA**（同事 3 个 PR 全部因此被关未合并，见 §4 教训）；issue 先查重 |
| **开源 issue（verl-project/verl-recipe）** | deepeyes 配方仓自身问题 | 同事 #166/#167 open 中，相关改动先评论别另开 |
| **开源 issue（vllm-project/vllm-ascend）** | NPU 推理引擎层问题 | **同事从未提过，蓝海**：如 AOT 缓存加载失败（T-#18） |
| **DTS 单** | 内部问题：文档纠错、驱动/HDK 缺陷（142 HDK 25.2.0 无 graph/IPC）、环境问题、内部工具缺陷 | 不适合开源的都走这里 |

## 2. 去重基线（同事 super-yuki 已提交，2026-11-08 查证）

### verl-project/verl
| 编号 | 类型 | 主题 | 状态 |
| --- | --- | --- | --- |
| #8550→#8559 + PR #8551→#8560 | bug+fix | Qwen3 XML tool parser 对 `...` 参数抛 Ellipsis | **#8560 open（未合并）**，其余关闭 |
| #8554 + PR #8555 | bug+fix | RLHFDataset 不解析相对 tool_config_path | issue closed，**PR 关闭未合并（CLA）** |
| #8552 + PR #8553 | bug+fix | 超长 prompt 过滤每次启动重跑（datasets fingerprint） | issue closed，**PR 关闭未合并（CLA）** |

### verl-project/verl-recipe
| 编号 | 类型 | 主题 | 状态 |
| --- | --- | --- | --- |
| #166 (PR) + #167 (issue) | feature | deepeyes Qwen3.5-4B/9B Ascend NPU 支持 | **均 open** |

### 引用过的外部修复（非同事提交）
- verl PR #8413（jagged position_ids）、pytorch#153238、vllm-ascend PR #6033/#5725（EI0020 端口）

**⚠️ 关键未决**：#8560 未合并 → 本地 tool_parser 补丁仍必须保留；#8553/#8555 未合并 → 配方侧
变通（prefilter 预过滤 + 绝对路径）仍必须保留。跟进合并状态也是工作项（§5）。

## 3. 挖掘方法（每类问题的找法）

1. **代码 review**：同事的 deepeyes/ 配方逐文件审（deepeyes.py / tools/ / scripts/ / configs/ / run_*.sh），
   重点对照案例团队文档里的经验性修复是否被保留（本次已发现 I-2）；
2. **文档 vs 实测**：同事记录 vs 机器实测（本次已发现 judge 模型记录缺失，I-3）；
3. **日志异常模式**：训练日志里的 WARNING/ERROR 模式统计（如 judge 400、EI0020、torch.compile 重编）；
4. **指标语义缺口**：监控指标在开关变化后的行为（如 bypass_mode 后 entropy 消失，I-4）；
5. **未提交的"待处理"**：同事文档里明确写"待处理/建议"的点（judge prompt 截断 I-1 即此来源）；
6. **蓝海仓库**：vllm-ascend / transformers / pytorch 上游，同事零提交的地方优先。

## 4. 候选问题池（状态：候选→已验证→已提交→已合并/关闭）

| ID | 类别 | 描述 | 证据 | 状态 | 建议渠道 |
| --- | --- | --- | --- | --- | --- |
| **I-1** | 代码缺陷+改进 | **judge 400 → 奖励污染（2026-11-11 二次定因：调用顺序缺陷，非上下文容量）**：`compute_score` 第 277 行先调 `_judge_accuracy`、第 280 行才做 `len(answer_text)≥1000 → acc=0` 判定——顺序反了。400 全部来自"缺 `<answer>` 标签"的格式错误 rollout（回退分支把整条 rollout 尾部可达 16384 token 当 answer 发 judge，超 8192 必 400），而这类样本奖励本就定型为 0，judge 结论从不影响奖励 → 每次 400 都是纯无效调用（3 次重试 + 万级 token prefill）。实测 87 次/2048 rollout ≈ 4%，与格式错误率一致。**原"按 token 预算截断 answer_text/GT"方案已推翻**：用 judge 自家 tokenizer 实测合法输入最坏 1760 token（预算 8176 的 4.6 倍余量），合法流量碰不到上限；截断 GT（数据集最长 1884 字符）反而会把 CORRECT 判成 INCORRECT 造假阴性。顺带：4xx 确定性失败仍重试 3 次（sleep 1s+2s）；失败日志只有状态码不含 vLLM 响应体（首次误判的直接原因） | judge 400 计数 89（T-#12 全记录）+ tokenizer 逐集实测 + KV cache 2,049,536 token（8192 仅用 0.4%）+ deepeyes.py:277-280 顺序确认 | **修复已落地**：本地 commit 9c56e7f（0 分判定提前到 judge 调用前 + 日志带 body/prompt_chars，4 个单测过，奖励数值逐位不变）→ judge max_model_len 32768 脚本已备不重启 → **可提 PR**（md 已落盘：`upstream/可提交_PR_verl-recipe_judge调用顺序修复与失败日志增强.md`） | verl-recipe PR + DTS-002（已更新为二次定因版） |
| **I-2** | 代码缺陷（回归） | **tool_reward 计入失败的工具调用（丢了防 reward-hacking 检查）**：`compute_score` 的 `has_tool_usage` 只查 `<tool_call>`/`<tool_response>` 标签存在（deepeyes.py:261-265）。bbox 非法时工具执行失败，但错误响应同样渲染为 `<tool_response>` → 仍得满分工具奖励。案例团队（同是 Qwen3.5）为此专门加过 `"Zoomed in on the image to the region (relative:" in solution_str` 成功标记检查（案例文档 §2.4 明确记录 reward hacking 现象），同事版本丢失该防线 | 代码 grep 确认无 "Zoomed in" 检查 + 案例文档 §2.4 记录 | **已解决（2026-11-12）**：本地 commit 894fef4 落地 TOOL_SUCCESS_MARKER 门控 + tool_used（真实成功，奖励相关）/tool_called（尝试调用，诊断用）分离；容器内 3 用例验收通过（失败调用 tool_used=0、成功 tool_used=1、无工具全 0）→ **可提 PR**（md：`upstream/`）。实轨迹补充不再必要：验收用例直接构造非法 bbox 轨迹调 compute_score，实测通过 | verl-recipe PR（关联 #166/#167）+ DTS-003（已升级为已解决版，含 3 图证据） |
| **I-3** | 内部文档错误 | 同事 TROUBLESHOOTING「环境」表未记录 judge 真实模型/部署形态（实际 Qwen3-30B-A3B-Instruct-2507、TP=8 占满 142 全部 8 卡、max_model_len 8192）；RUNBOOK §6.1 示例写 Qwen2.5-32B 会误导重建；「单步耗时分析」的 79-92 分钟是 142 语境，与 138 实测 29 分钟差异未标注 | 142 实测 + 本机 run_full_138.sh | 已验证 | DTS / 直接在 archted 仓 wiki 修正 |
| **I-4** | 监控/文档 | `algorithm.rollout_correction.bypass_mode=True` 后 `actor/entropy` 指标不再产出（TROUBLESHOOTING 提速实验实测），监控面板/告警若依赖该指标会误判 | TROUBLESHOOTING「提速实验」+ 代码（calculate_entropy=False 时不计算） | 候选（需确认上游是否已知/有 issue） | verl issue（先查重）或文档 PR |
| **I-5** | 推理引擎 | vllm-ascend AOT compile 缓存产物（含 inductor Triton wrapper）在 Ascend 无法加载，每次启动每 worker 重编 1-2 分钟并刷 WARNING（T-#18） | TROUBLESHOOTING #18 | 候选 → 需收集完整日志片段 | **vllm-ascend issue（蓝海，同事零提交）** |
| **I-6** | 工程健壮性 | 同事 `#8555/#8553` 被关未合并后，配方侧依赖变通（绝对路径 + prefilter），若他人按 README 直接跑会踩同样坑——可在 #167 下评论提醒或在 README 补充说明 | PR 关闭记录 | 候选 | verl-recipe（评论/文档 PR） |
| **I-9** | 训练健壮性（高价值） | **GRPO 零方差奖励 → 训练静默死亡，无告警无中止**：正式 run 实测（E-1）——step 184 起 `critic/score/mean` 恒 -0.2、`advantages≡0`、`pg_loss=0`、`grad_norm=0`，训练循环不报错继续跑了 100+ 步（~50 小时 × 8 卡白白空转，ETA 还剩 109h）。val 的 acc=0/tool_used=0 佐证模型输出已塌缩。verl 应支持：连续 N 步 advantage 全零（或 score 恒等于某常数）时告警/中止/回滚到上一 ckpt。附带：bypass_mode 下 entropy 不产出（I-4）加剧了无监控 | 日志 `full_outer_138_v3.log` step 184-291 + `eval_prep/metrics_report/1_score.png` 悬崖图 | 已验证 → **可提 issue（强证据链）**，PR 可做"全零 advantage N 步熔断" | verl issue + DTS |
| **I-8** | 上游配方脚本缺陷 | **verl-recipe 上游 `recipe/deepeyes/run_deepeyes_grpo.sh` 的 `REF_MODEL_PATH` 是死占位符**：第 16 行 `REF_MODEL_PATH=ref_model_path`（字面量，从未赋真实值），第 30 行用作 `actor_rollout_ref.model.path=${REF_MODEL_PATH}` → 照原样运行必然报"路径不存在"。上游 commit 21892b9（migrate recipes from verl #7）带入，非同事改动。修法：改为 `REF_MODEL_PATH=${REF_MODEL_PATH:?set to your base model dir}` 或在 README 标注必填 | `git log -- deepeyes/run_deepeyes_grpo.sh` + 第 16/30 行 | 已验证 → 小 PR（低优先级，但稳定可收） | verl-recipe issue/PR |
| **I-10** | 驱动/HDK 缺陷 | **NPU 驱动按命名空间独占设备，新容器 -8020 无任何指引（E-2）**：旧容器占用 NPU 期间，同参数新建容器 `npu-smi info` 报 `device is used. ret is -8020`，与挂载参数/镜像/SELinux/cgroup 均无关；真因是驱动 UDA 按命名空间独占（dmesg `uda_occupy_dev_by_ns: Conflict open udevid` errno 87），报错文本与官方文档均不提示，唯一绕过是 --privileged（不可作方案） | dmesg uda_occupy_dev_by_ns 实锤 + 逐项 bisect 记录（TROUBLESHOOTING E-2） | 已验证 → **可提交** | DTS-005 |
| **I-11** | 驱动缺陷 | **NPU 进程退出后驱动级资源残留（脏卡）→ Too many streams（E-6）**：进程正常退出后部分卡残留 ~3.4G/卡无主 HBM + 流表脏状态（npu-smi Process 表查不到进程），新进程在该卡初始化稳定报 `Alloc Stream resource failed: Too many streams are created`，与显存余量无关；唯一恢复 = 重启容器或换卡。报错完全不指向资源残留 | npu-smi 占用/进程表比对 + 换卡对照实验（TROUBLESHOOTING E-6） | 已验证 → **可提交** | DTS-006 |
| **I-12** | 环境基线缺陷 | **宿主机 /etc/hosts 缺 localhost 条目 → 容器内 Ray 初始化全挂（E-3）**：openEuler 22.03 宿主机 hosts 被精简（无 `127.0.0.1 localhost`），Docker 容器默认继承 → 容器内 `gethostbyname("localhost")` gaierror，Ray 报 `Could not find any running Ray instance`（与真因相距甚远）；`--add-host localhost:127.0.0.1` 即愈 | getent hosts 空 + --add-host 对照（TROUBLESHOOTING E-3） | 已验证 → **可提交** | DTS-007 |
| **I-13** | 内部工具缺陷 | **docker exec -d 长命令偶发静默失败（E-7 前半）**：`docker exec -d` 派发长启动命令偶发"返回 0 但什么都没发生"（无日志、无进程、无报错），138/142 两机均复现，同命令重试可成功；规避模式 = 落盘脚本再 exec 或 bash -c 内自管 nohup | 两机复现记录 + 重试对照（TROUBLESHOOTING E-7） | 已验证 → **可提交** | DTS-008 |
| **I-14** | 推理引擎 | **torch 2.10 默认开启 AOT → vllm-ascend 启动必崩（E-4）**：vLLM 对 torch≥2.10 默认 `VLLM_USE_AOT_COMPILE=1`，vllm-ascend 不支持 AOT 路径 → 0.27.1rc1 镜像开箱即崩（`aot_compile is not supported`）；绕过 `VLLM_USE_AOT_COMPILE=0`（ray runtime_env 里必须传字符串 `'"0"'`，int 0 会撞无关的 TypeError）。蓝海：同事在 vllm-ascend 零提交 | 完整 traceback + 绕过后 PIECEWISE capture 35/35 正常（TROUBLESHOOTING E-4） | 已验证 → **可提交** | vllm-ascend issue（蓝海） |
| **I-15** | 代码缺陷（默认值） | **工具配置默认绝对坐标 vs Qwen3.5 相对坐标 + TOOL_NAME 不一致静默失败（E-11）**：①配方默认 `image_zoom_in_tool_config.yaml`（绝对像素），Qwen3.5 输出 0-1000 相对坐标 → 裁剪错乱无报错；②tool_config_path 与数据集 TOOL_NAME 不一致时**每个工具调用都静默失败**：训练跑完 tool_used=0、tool 奖励通道恒 0，无任何告警（P0-1 审计发现，损失一整轮训练）。修法：默认换 `image_zoom_in_rel_tool_config.yaml`（已在 yaml:36 + 启动脚本:165 落地，score/max 触 2.0 上限验证）+ 启动时校验一致性 fail fast | 案例一致性审计 + 修复后训练 score/max=2.0 证据（TROUBLESHOOTING E-11） | 已验证 → **可提交**（issue + 校验 PR） | verl-recipe issue + PR（关联 #166/#167） |
| **I-16** | 数据质量 | **数据集混入无 ground_truth 样本不告警**：训练集 33,137 条中 thinklite 子集 10,953 条（~1/3）无可用 GT，永远拿不到 acc 奖励却消耗全部 rollout/judge/训练算力，并稀释 GRPO 组内基线；无 GT 样本只能挣 format+tool 分，最省事的得分方式是短格式输出——与 policy collapse（短输出模式）疑似相关。prepare_data 打印无 GT 占比 + `--vstar-only` 过滤已实现 | prepare_data 统计（10953/33137）+ 本地 --vstar-only commit a7c10a2 | 已验证 → **可提交**（issue + PR 代码已就绪） | verl-recipe issue + PR |
| **I-17** | 配置矛盾 | **run 脚本强制 `TORCHDYNAMO_DISABLE=1` 与 `use_torch_compile=True` 冲突，compile 静默失效**：`run_deepeyes_grpo_qwen3_5.sh:64` 无条件 export，进程级禁 Dynamo，用户以为 compile 开着实际全程 eager；41 步日志零 dynamo/inductor 输出，无任何告警。与实测 MFU 1.6%/AICore ~30%（dispatch-bound）自洽 | run_deepeyes_grpo_qwen3_5.sh:64 + run_full_v1.sh:81 + 日志 | 已验证 → **可提交**（issue + PR <10 行） | verl-recipe issue + PR |
| **I-18** | 内部工具缺陷 | **opencode v1 风格 `maxOutputTokens` 配置被静默忽略（E-9）**：opencode v2.2.5 schema 用 `limit.{context,output}`，v1 风格键无校验无告警直接忽略 → 模型默认 completion 预算 32000，对严格后端（judge vLLM max_model_len=8192）每请求必败 `max_completion_tokens=32000 cannot be greater than max_model_len=8192`，而配置里明明写了 8192；对宽松网关则不可见，同配置一边能用一边崩 | 容器内 opencode + judge 实测报错 + 配置比对（TROUBLESHOOTING E-9） | 已验证 → **可提交** | opencode issue |
| **I-19** | 代码缺陷 | **tensordict `index_select_tensor_dict` 在 unbind 产出含 Ellipsis 叶子时失败**（本地补丁②）：jagged nested tensor unbind 出 `...` 哨兵叶子，下游 JSON 序列化失败。本地 fallback 从 values/offsets 重建。需查重（同事 #8550/#8559 只覆盖 tool_parser 侧） | 本地补丁② + 失败现场 | 待提交（需 5 行最小复现 + 查重 + main 复现确认） | verl issue |
| **I-20** | 性能（配方默认值） | **`ppo_micro_batch_size_per_gpu=1` 默认值致 actor update latency-bound**：9B 8×910B3 实测 MFU 1.6%、AICore 均值 ~30%（峰 100% 谷 ~1%）、update_actor 18.3min/24-26min 步——kernel launch 数随 micro-batch 数而非 token 数增长；案例团队同模型族用 micro_batch=8，配方默认 1 是偏离 | profiling 证据链（MFU/AICore/耗时分解） | 待提交（**等 micro 1→8 SMOKE 前后实测数据**回填后提） | verl-recipe issue / #167 评论 |
| **I-21** | 内部工具缺陷 | **opencode 管理服务端口固定 49374 无自动回退（E-8）**：--net=host 容器内跑 opencode 与宿主机 opencode 撞固定端口 49374，需手工 `opencode service set port 49555`；建议支持 env（OPENCODE_SERVICE_PORT）或自动回退下一空闲端口 | 容器/宿主机同跑冲突实测（TROUBLESHOOTING E-8） | 待提交（查重 opencode 仓端口机制） | opencode issue |
| **I-22** | 训练健壮性 | **checkpoint 滚动覆盖（max_actor_ckpt_to_keep）致崩溃前健康权重不可恢复（E-1 衍生）**：step 184 collapse 后 ckpt 滚动覆盖，发现时（step 292）保留的 ckpt 全是塌缩区权重，最后健康权重（step 170）已被删 → 只能从头重训（~9 天单机）。修法：异常信号激活时暂停删除 / best-k-by-val 选项 / 文档明示风险 | E-1 崩溃时间线 + ckpt 保留清单 | 已验证 → **可提交** | verl issue |
| **I-23** | DX | **batch 整除断言报错不含约束来源与修法指引（E-7 后半）**：`real_train_batch_size (64) must be divisible by minimal possible batch size (N)` 不说明 N 如何推导（world size/DP）、不列合法 batch 取值、多机场景 NDEVICES_PER_NODE 常见坑无提示；6 卡冒烟 batch 64 直接 bisect 卡数 | 142 冒烟实录（TROUBLESHOOTING E-7） | 已验证 → **可提交** | verl issue |
| **I-7** | 待挖掘 | deepeyes.py 其余逻辑（answer 抽取回退链、1000 字符惩罚）、tools/image_zoom_in_tool.py、run_*.sh、prefilter 脚本的逐行 review；`run_deepeyes_grpo_qwen3_5.sh` 的 PYTHON 解析器问题（T-#1 相关）是否已修 | 待做 | 待办 | — |

## 5. 提交台账（我提交的记录）

### 5.0 DTS 问题单文档（已备好待提交，2026-11-10）

> 文档在本地 `deepeyes/pr_issue/`（**已加 .gitignore，永不提交 GitHub**），按 `pr_issue/DTS问题单模板.docx`
> 字段结构生成（问题现象/操作步骤/预期结果/实际结果/原因定位/修复建议），提交后回填单号到下表：

| 本地文档 | 对应问题 | 截图 | 状态 |
| --- | --- | --- | --- |
| `DTS-001_E1-I9_GRPO零方差奖励致训练静默死亡.docx` | E-1 / I-9（合并一单） | 嵌入 4 图：1_score/3_grad_norm/4_kl 指标图 + 日志零方差恒值 grep 实拍（step:100 健康对照 vs step:184 恒 -0.2，计数 112） | ✅ 可提交 |
| `DTS-002_I1_judge评分prompt超长400致奖励污染.docx` | I-1（**2026-11-12 已更新为二次定因版：调用顺序缺陷 + 0 分判定提前 + 4xx 不重试；原截断建议已删除**） | 嵌入 7 图（全部终端命令实拍，图内首行即复现命令）：400 日志统计与原文（90 次）、vLLM reason 文档原文、修复前代码（git 9c56e7f~1）、diff 判定前移、diff 日志增强、judge 脚本 MAX_MODEL_LEN=32768、容器验收 | ✅ 可提交 |
| `DTS-003_I2_tool_reward计入失败工具调用可被hacking.docx` | I-2（**2026-11-12 升级为已解决：commit 894fef4 成功标记门控 + tool_used/tool_called 分离，容器内 3 用例验收通过**） | 嵌入 3 图（终端命令实拍）：修复前代码（git 894fef4~1）、commit 894fef4 diff、容器内验收实测（失败调用 tool_used=0） | ✅ 可提交 |
| `DTS-004_I3_交接文档judge部署信息缺失误导重建.docx` | I-3 | 嵌入 1 图：RUNBOOK judge 行 grep + 训练日志 judge 端点提取实拍 | ✅ 可提交 |
| `DTS-005_E2-I10_NPU驱动按命名空间独占设备新容器8020无指引.docx` | E-2 / I-10 | 嵌入 1 图：NPU 占用现状实拍（训练容器运行中、八卡 AICore 100%）；-8020/dmesg 为现场记录 txt | ✅ 可提交 |
| `DTS-006_E6-I11_NPU进程退出后驱动级资源残留脏卡TooManyStreams.docx` | E-6 / I-11 | 嵌入 1 图：Too many streams 日志 grep 实拍（含上下文块） | ✅ 可提交 |
| `DTS-007_E3-I12_宿主机hosts缺localhost条目容器内Ray初始化失败.docx` | E-3 / I-12 | 嵌入 1 图：/etc/hosts 缺条目 + 生产容器 ExtraHosts 修复参数 + 容器内解析现状三连实拍（2026-11-12） | ✅ 可提交 |
| `DTS-008_E7-I13_docker_exec-d长命令偶发静默失败无诊断.docx` | E-7 / I-13 | 无图（偶发现象无落盘；现场记录 txt 摘录见 evidence/） | ✅ 可提交 |

> **证据图目录（2026-11-12 建立，同日升级为终端实拍式）**：`pr_issue/evidence/` 存有 16 张 PNG + 对应 .txt 摘录。
> PNG 均为**真实终端实拍式**：在真实 tmux 终端里执行真实命令（bash 原始提示符 + 命令原始 ANSI 颜色，无任何加工），
> `capture-pane` 抓取真实屏幕后 1:1 渲染——**图内第一行即复现命令**，可在同机直接重跑验证。
> 例外：DTS-001 前三张为 matplotlib 指标图；DTS-005/DTS-008 的现场记录（当时无落盘）只提供 .txt 摘录并明确标注，不冒充实拍。
> 每张 PNG 同名 .txt 为同内容文本摘录，供人工重新截图。
> 重新生成：`python3 /home/h00516332/tmp_scripts/capture_render_evidence.py`（tmux 实拍 + 渲染一体）。

> **md 镜像规则（2026-11-12 起）**：每份 docx 自动派生 `可提交_DTS-00x.md` 到同目录，方便快速复制粘贴；
> docx 是提交载体、md 是复制载体，内容以 docx 为准。
> 重新生成：`python3 /home/h00516332/tmp_scripts/gen_dts.py`（docx）→
> `python3 /home/h00516332/tmp_scripts/gen_dts_md_mirror.py`（md 镜像，含处理状态）。
>
> **字段模板（2026-11-12 更新）**：基础 6 字段【问题现象】【操作步骤】【预期结果】【实际结果】【问题根因】【修复建议】；
> **处理状态为"已解决"的单追加 3 字段【解决思路】【验收环境】【验收结果】**（当前：DTS-002/003/004/005/006/007/008；
> DTS-001 框架熔断未实现，暂为待修复状态，验收后补段）。每张单的处理状态同步记录在 md 镜像文件头。
> DTS-003 已于 2026-11-12 升级为已解决（commit 894fef4 落地 + 容器内 3 用例验收）。

> **2026-11-12 重生成（修中文乱码）**：4 份 docx 打开时中文乱码，根因是 `gen_dts.py` 只设了
> Calibri（python-docx 的 `font.name` 只写 `w:ascii/w:hAnsi`），theme 的东亚字体（`<a:ea>`）又是
> 空串 → 中文 run 无字体可解析。已在 `save_doc()` 里给所有 run 与样式显式补 `w:eastAsia="宋体"`
> 并重新生成（内容一字未动，逐 run 校验 0 个含中文 run 缺字体声明，DTS-001 三张图完好）。
> 后续再用 `gen_dts.py` 生成新单自带此修复。

（截图独立文件 1_score.png / 3_grad_norm.png / 4_kl.png 也在 pr_issue/ 备份，DTS 系统若需单独传图可直接取用。）

### 5.0.1 开源 issue/PR md 承载文件索引（`deepeyes/upstream/`，随仓提交，2026-11-12 建立）

> 每个文件头含渠道/状态/台账 ID/提交前置检查，正文为可直接复制粘贴的英文稿。
> 状态变化时**改文件名前缀**（可提交/待提交）并同步本台账。生成脚本：`/home/h00516332/tmp_scripts/gen_upstream.py`。

**可提交 issue ×11：**

| 文件 | 台账 ID | 渠道 |
| --- | --- | --- |
| `可提交_ISSUE_verl_GRPO零方差奖励静默死亡无熔断.md` | I-9 / E-1 | verl |
| `可提交_ISSUE_verl_bypass模式下actor-entropy指标静默消失.md` | I-4 | verl |
| `可提交_ISSUE_verl_checkpoint滚动覆盖导致崩溃前健康权重不可恢复.md` | I-22 | verl |
| `可提交_ISSUE_verl_batch整除断言报错不含卡数指引.md` | I-23 / E-7 | verl |
| `可提交_ISSUE_vllm-ascend_torch2.10默认开启AOT导致启动必崩.md` | I-14 / E-4 | vllm-ascend（蓝海） |
| `可提交_ISSUE_verl-recipe_工具配置与Qwen3.5相对坐标及数据集TOOL_NAME不一致静默失败.md` | I-15 / E-11 | verl-recipe |
| `可提交_ISSUE_verl-recipe_数据集混入无ground_truth样本不告警.md` | I-16 | verl-recipe |
| `可提交_ISSUE_verl-recipe_run脚本导出TORCHDYNAMO_DISABLE与use_torch_compile矛盾.md` | I-17 | verl-recipe |
| `可提交_ISSUE_verl-recipe_上游run_deepeyes_grpo.sh的REF_MODEL_PATH死占位符.md` | I-8 | verl-recipe（正文在 drafts/I8_issue_verl-recipe.md） |
| `可提交_ISSUE_verl-recipe_tool_reward计入失败的工具调用.md` | I-2 | verl-recipe |
| `可提交_ISSUE_opencode_v1风格maxOutputTokens字段被静默忽略.md` | I-18 / E-9 | opencode |

**可提交 PR ×4：**

| 文件 | 台账 ID | 代码就绪度 |
| --- | --- | --- |
| `可提交_PR_verl-recipe_judge调用顺序修复与失败日志增强.md` | I-1 | ✅ commit 9c56e7f |
| `可提交_PR_verl-recipe_REF_MODEL_PATH死占位符修复.md` | I-8 | ✅ fork 分支 bd82d24 |
| `可提交_PR_verl-recipe_TORCHDYNAMO_DISABLE冲突修复.md` | I-17 | 待打分支（<10 行） |
| `可提交_PR_verl-recipe_prepare_data增加vstar-only过滤开关.md` | I-16 | ✅ commit a7c10a2 |

**待提交 issue ×5：**

| 文件 | 台账 ID | 补什么 |
| --- | --- | --- |
| `待提交_ISSUE_vllm-ascend_AOT缓存产物每次启动重编.md` | I-5 / T-#18 | 完整 WARNING 块 + 时间戳 |
| `待提交_ISSUE_verl_index_select_tensor_dict_unbind失败还原需查重.md` | I-19（补丁②） | 查重 + 5 行最小复现 |
| `待提交_ISSUE_verl-recipe_micro_batch默认1导致latency-bound待SMOKE数据.md` | I-20 | micro 1→8 SMOKE 前后数据 |
| `待提交_ISSUE_verl-recipe_README补充绝对路径与prefilter变通说明.md` | I-6 | 决定评论 or 文档 PR |
| `待提交_ISSUE_opencode_服务端口固定49374无自动回退.md` | I-21 / E-8 | 查重 + 复现步骤 |

**待提交 PR ×2：**

| 文件 | 台账 ID | 前置 |
| --- | --- | --- |
| `待提交_PR_verl-recipe_tool_reward成功标记门控修复.md` | I-2 | 先提 issue + 抽实轨迹 + 签 CLA |
| `待提交_PR_verl-recipe_工具配置与数据集TOOL_NAME一致性校验.md` | I-15 / E-11 | 写校验代码 + 先提 I-15 issue |

> 模板：`| 日期 | 渠道 | 编号/链接 | 内容 | 状态 |`
> **PR 流程**：①fork ②**签 CLA**（cla-assistant.io/verl-project/verl，用 archted 账号）③分支基于最新 main ④改动最小化+带复现证据 ⑤关联同事 open 的 #167（配方类）避免重复开题

| 日期 | 渠道 | 编号 | 内容 | 状态 |
| --- | --- | --- | --- | --- |
| （暂无） | | | | |

**准备中（2026-11-08）：**
- I-8 issue 文稿：`deepeyes/drafts/I8_issue_verl-recipe.md`（已查重：上游无人提过；#4989 是 deepeyes.py system prompt，不撞车；上游 main 至今未修）
- I-8 PR 分支：fork `archted/verl-recipe` 分支 `fix-deepeyes-run-script-model-path`（commit bd82d24，基于上游 main，`bash -n` 语法已验）
- **PR 创建方式（2026-11-08 实测）：细粒度 token 未授权官方仓 PR 写权限，API 创建 403；改用预填 compare 链接手动创建——链接全文在 `deepeyes/drafts/I8_pr_create_url.txt`（已含标题+正文，打开点 "Create pull request" 即可）。archted 已签 CLA。issue 等用户指令再发（用户要求先 PR 后 issue，防抢跑）**
- DTS 单草稿 ×2：`deepeyes/drafts/DTS_drafts.md`（I-8 联动单 + I-3 文档纠错单），DTS 需 archted 手动提交

## 6. 教训库（如何让 PR 被收）

1. **先签 CLA 再提 PR**——同事 3 个 PR（#8551/8553/8555）全部因 CLA 未签被关闭未合并，工作白做；
2. 提交前在 issue 区先发 issue 等维护者反馈（同事 #8550→#8559 重提过一次，说明初次 issue 可能被关）；
3. 证据链：现象日志 + 根因分析 + 最小复现 + 修法，附在本机实测数据；
4. 每个已验证候选先写成 issue 草稿存本文件 §4，再动手提交；
5. 已合并的修复要跟进同步回本地仓库（删对应本地补丁/变通配置）。

## 7. 跟进清单

- [ ] 盯 #8560（tool_parser `...` 修复）合并状态，合并后删除本地 tool_parser 补丁
- [ ] 盯 #166/#167（Qwen3.5 NPU 支持）review 意见并响应
- [x] I-1 judge 400：二次定因 + 修复落地（commit 9c56e7f）+ DTS-002/PR md 已更新为二次定因版（2026-11-12）
- [x] I-10~I-23 建档 + 22 份 upstream md 落盘（2026-11-12）
- [x] I-2 修复落地（commit 894fef4）+ 容器内 3 用例验收（失败调用 tool_used=0）+ DTS-003 升级为已解决版含 3 图证据（2026-11-12）
- [x] DTS-001~008 证据图升级为终端实拍式（capture_render_evidence.py：tmux 真实终端执行真实命令 → capture-pane 抓屏 → 1:1 渲染，图内首行即复现命令；撤掉渲染风图与因果不成立的 smoke Ray 日志图，现场记录类仅留 txt）+ md 镜像同步（2026-11-12）
- [x] I-2 实轨迹证据需求已关闭：容器内直接构造非法 bbox 轨迹调 compute_score 实测验收（3 用例），效果等同且更直接（2026-11-12）
- [ ] I-4/I-5 在上游查重后决定是否提（I-4 md 已就绪，I-5 待补日志）
- [ ] I-8 用 drafts/I8_pr_create_url.txt 预填链接手动创建 issue + PR（archted 已签 CLA）
- [ ] I-20 等 micro 1→8 SMOKE 数据回填后提 issue
- [ ] 可提交 md 批次提交顺序建议：先 issue 后 PR（防抢跑）；配方类先在 #167 评论
- [ ] 每周复盘新增候选；**新问题发现后当天落盘 md（upstream/ 或 gen_dts.py），同步本台账**
