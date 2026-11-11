# ARCHTED Troubleshooting · 我的问题归档

> 维护人：archted
> 创建：2026-11-08
> 用法：**每次遇到异常先在这里搜；解决了就按模板登记一条**。同事的 19 个历史问题见
> 本目录 `QWEN3_5_NPU_TROUBLESHOOTING.md`（细节以原文为准，下表只是速查索引）。

---

## A. 同事已知问题速查索引（详情看原文档）

| # | 现象关键词 | 根因一句话 | 处理一句话 |
| --- | --- | --- | --- |
| 1 | Ray 报 Python 版本不匹配 | Ray 与驱动解释器版本不一致 | 统一用 conda 解释器启动 Ray |
| 2 | `No module named 'acl'` | conda env 缺 CANN Python 路径 | `image.pth` 追加 toolkit python site-packages |
| 3 | `hcclCommInitRootInfoConfig error code 1` | 多网卡 HCCL 选错 | `HCCL_SOCKET_IFNAME=<业务网卡>`，写进 Ray runtime_env |
| 4 | `EI0020 ... port 16666 already bound` | 多 vLLM 副本 NPU socket 端口冲突 | `HCCL_HOST/NPU_SOCKET_PORT_RANGE` 写进 runtime_env |
| 5 | 数据过滤时工具配置文件找不到 | 相对路径在 Ray actor cwd 失效 | `data.tool_config_path` 用**绝对路径**（上游 #8554） |
| 6 | `No module named 'qwen_vl_utils'` | 缺依赖 | `pip install qwen-vl-utils==0.0.14 --no-deps` |
| 7 | 退出码 134 / glibc abort | conda + 镜像扩展退出堆破坏 | 训练完成后 abort 无害，忽略 |
| 8 | `capture_end error 507903` / plog 207005 | HDK 25.2.0 < CANN 9.1.0 配套驱动 | `enforce_eager=True`；根治=升级 HDK ≥25.5 |
| 9 | `FRACTAL_NZ mode is enabled` | vllm-ascend 0.27.1 新配置键 | `+...engine_kwargs.vllm.additional_config.weight_nz_mode=0` |
| 10 | `'FSDPQwen3_5DecoderLayer' has no attribute 'layer_type'` | transformers 5.14.1 改名 | 固定 `transformers==5.12.1`（verl 要 <5.13） |
| 11 | `split_with_sizes expects ... got [4]` | position_ids 3D jagged unbind bug | **verl 核心补丁 #8413**（不在本仓！见 §C） |
| 12 | judge 客户端 60s 超时（14h 内 184 次） | judge 长尾生成占住推理 | judge 加 `--override-generation-config '{"max_new_tokens": 64}'`；副作用：prompt>8128 的 judge 请求 400（约 8 次/步记 0 分），待修 `_judge_accuracy` 截断 |
| 13 | vLLM 报 Free memory 4.29 GiB | 其他容器占满卡 | 宿主机 `docker ps` + `npu-smi` 排查，找资源方停容器 |
| 14 | 重启后显存不足、残留 `VLLMWorker_TP` | 引擎失败 worker 残留 | 按 PID 清残留进程再启动 |
| 15 | 每步 rollout 后 host 空转 ~27 分钟 | HDK<25.3 无 IPC，权重同步退化为共享内存 | 升级 HDK 或迁 138 |
| 16 | 每次启动重复过滤 13–16 分钟 | datasets.filter fingerprint 不命中 | 预过滤生成 `*.filtered_8192.parquet` + `filter_overlong_prompts=False`（脚本 `scripts/prefilter_deepeyes.py`；上游 #8552） |
| 17 | CPU actor 报 PlatformNPU not available | 无 NPU actor import vLLM 的平台探测噪声 | 无害，忽略 |
| 18 | torch.compile 缓存加载失败重新编译 | Ascend 无法加载 AOT 缓存产物 | 预期回退，启动多花 1–2 分钟，无需处理 |
| 19 | `Object of type ellipsis is not JSON serializable` | 模型输出 `...` 被 literal_eval 成 Ellipsis | **verl 核心补丁**（tool_parser 保留原字符串，见 §C） |

**性能经验**（同事实测，正式训练单步 79–92 分钟的构成）：
rollout 40–50min ＋ 权重同步 25–30min（IPC 缺失）＋ actor 更新 5–10min。
`algorithm.rollout_correction.bypass_mode=True` 确定收益约 220s/步（-11%）；
`actor.use_torch_compile` 在本任务无收益（entropy 不计算）。详见原文档「提速实验」。

**复现所需全套开关**（同事归纳）：
- 数据：prefilter + `filter_overlong_prompts=False` + `data.tool_config_path` 绝对路径
- 训练：`enforce_eager=True`（HDK 旧时）+ `weight_nz_mode=0`
- judge：`max_new_tokens=64`
- conda：`transformers==5.12.1`、`qwen-vl-utils==0.0.14 --no-deps`
- verl 核心补丁 ×3（见 §C）

---

## B. 我遇到的问题日志（按时间倒序追加）

### E-11｜`tool_config_path` 与数据集 `TOOL_NAME` 不一致 → 工具调用全部静默失败（2026-11-11，案例一致性审计）

- **现象**：直接执行 `bash recipe/deepeyes/run_deepeyes_grpo_qwen3_5_9b.sh` 时训练**不报错也能跑完**，但 `critic/score/mean` 长期停在 ~0.8 以下（=0.8×acc+0.2×format，工具奖励 1.2 永远拿不到），日志里会刷 `Unknown function '...'` 或 `Error executing tool '...': Missing required 'image' parameter in kwargs`。
- **根因链**：
  1. 数据集侧 `deepeyes.py` 已把工具名改成 `image_zoom_in_rel_tool`（案例 §2.4 要求，相对坐标），`tools_kwargs` 的键也跟着变成 rel 名；
  2. 但 `configs/deepeyes_multiturn_grpo_qwen3_5.yaml` 与 `run_deepeyes_grpo_qwen3_5.sh` 默认仍指向 `image_zoom_in_tool_config.yaml`（绝对坐标工具，schema name = `image_zoom_in_tool`）；
  3. `tool_agent_loop.py:457` 按 `tool_call.name` 在 `active_tools` 里查名字 → 查不到直接返回错误文本；即便名字匹配，`_call_tool` 用 `tools_kwargs.get(tool_name, {})` 取 create_kwargs，键不匹配就是空 dict → `ImageZoomInToolRelative.create()` 抛 `Missing required 'image' parameter`，异常被 `except` 吞掉只记 warning。
- **为什么现在没炸**：正式 run 走 `run_full_v1.sh`，它显式覆盖了 `data.tool_config_path` 与 `actor_rollout_ref.rollout.multi_turn.tool_config_path` 都指向 rel 配置。**只有绕开 run_full_v1 才会踩**。
- **修复**：把 recipe 默认值改对（yaml + `run_deepeyes_grpo_qwen3_5.sh` 默认 rel 配置），不要依赖启动脚本覆盖。判别命令：`grep -n "tool_config_path" <yaml> <run.sh>` 必须全部命中 `image_zoom_in_rel_tool_config.yaml`。
- **自检**：跑 1 步后 `grep -c "Unknown function\|Missing required 'image'" <log>` 必须为 0。
- **状态**：✅ **已修（2026-11-11）**。`configs/deepeyes_multiturn_grpo_qwen3_5.yaml:36` 与 `run_deepeyes_grpo_qwen3_5.sh:165` 默认值已改为 `image_zoom_in_rel_tool_config.yaml`；rel 配置的 schema `name` = `image_zoom_in_rel_tool` = `deepeyes.py:34` 的 `TOOL_NAME` = 数据集 `tools_kwargs` 键，三者一致。下次启动生效（当前 run 由 `run_full_v1.sh` 覆盖，行为不变）。

### E-10｜judge 返回 400（prompt 超 max_model_len）→ 每次都是"奖励已定型"的无效调用（2026-11-11，案例一致性审计 + 二次定因）

- **现象**：训练日志出现 `Chat completion request failed: 400 Client Error: Bad Request for url: http://71.10.29.142:18901/v1/chat/completions`，本 run 截至 step 32 共 **87 次**（≈2048 个 rollout 的 4%）。

**第一次判断（错，已推翻）**："judge 部署的 `max_model_len=8192` 太小，答对的长答案被判 0"。据此做了 prompt 截断（question/GT/answer）。**这是治表象**：截断 GT 反而可能把 CORRECT 判成 INCORRECT（训练集 GT 最长 1884 字符，被截到 1000）。

**二次定因（有数字）**：

1. `answer_text` 不是整条 rollout：`compute_score` 先从 `<answer>…</answer>` 里抽（`deepeyes.py:252`），只有**没有 `<answer>`**（格式错误）时才回退成"最后一个 `</tool_response>` 之后的全部内容"或整条响应（`:256-270`）—— 长度受 `max_response_length=16384` 约束。
2. 用 judge 自己的 tokenizer（Qwen3-30B-A3B）实测**合法输入**：
   - 全训练集最长的 question+GT（116 + 1884 字符）= 394 token；
   - few-shot + system ≈ 751 token；
   - 999 字符中文答案（= 长度上限，见第 3 点）= 615 token；
   - **合法最坏情况合计 1760 token，距 8176 预算有 4.6 倍余量。**
3. `compute_score:280` 规定 `len(answer_text) >= 1000` 直接 `acc=0`（反 judge hack 规则）。所以能超 8176 token 的请求必然 `answer_text` > 1.3 万字符 —— 只可能来自第 1 条的回退分支，而那条分支**已经被判 0 分**，judge 的结论无论如何都不影响奖励。
4. 结论：**judge 上下文不是瓶颈**（vLLM 启动日志实测 KV cache 可容 2,049,536 token，8192 只用了 0.4%）。真正的缺陷是**调用顺序**：先花一次 judge 调用，再判断"这个答案按规则已经是 0 分"。
5. 副因（可诊断性）：`requests` 的 `HTTPError.__str__` 只有状态码，日志里看不到 vLLM 返回的 reason（"maximum context length is 8192 tokens…"），所以第一次只能靠猜。

- **影响**：不改变奖励（那 4% 本来就 acc=0 + format=-1），但每次白烧一次 judge 调用（含 1.6 万 token 的 prefill），并**污染日志、掩盖真正的信号**——4% 的 rollout 是格式错误（缺 `<answer>` / 标签不配对），这才是该盯的指标。
- **修复（2026-11-11，已落地）**：
  1. **把长度判定提到 judge 调用之前**：`len(answer_text) >= 1000` → `acc=0, format=-1`，不再调用 judge。400 归零、judge 负载降 ~4%，奖励数值完全不变。
  2. **删掉截断**（question/GT/answer 全部按原样送判），避免 GT 被截导致的假阴性。
  3. 失败日志补上 vLLM 的响应体与 prompt 字符数：`... failed: 400 ... body='{"error":{"message":"maximum context length is 8192 tokens"…}}' (prompt_chars=N, total K)`。
  4. judge 部署 `max_model_len` 8192 → 32768（`MAX_MODEL_LEN` 环境变量可覆盖）。这不是本次 400 的成因，但 8192 本就是抄来的默认值（KV 实测可容 200 万 token），提高它给未来长 GT 数据留余量；**脚本已改，下次 judge 重启生效**（当前不重启：对现数据零收益，白赚 3–4 分钟停机）。
- **验证**：容器内 4 个用例（`_test_judge_order.py`）：
  | 用例 | 结果 |
  | --- | --- |
  | 无 `<answer>`、5000 字符尾巴 | judge 调用 **0 次**，acc 0，format -1 |
  | 短答案 `<answer>yes</answer>` | judge 调用 1 次，acc 1 |
  | 1200 字符答案 | judge 调用 **0 次**，acc 0，format -1 |
  | judge 返回 400 | 不崩，日志含 `body=…` 与 `prompt_chars`，`_JUDGE_FAILURES=1` |
- **教训**：`grep -c` 出的错误条数是"症状计数"，不是"根因定位"。先量出**合法输入的分布**（这里 1760 vs 8176），再判断是"容量不够"还是"输入不该发"。容量类问题看 KV cache 实测值，不看配置默认值。

### E-9｜opencode 自定义模型 token 上限字段：v2 用 limit，v1 风格 maxOutputTokens 被静默忽略（2026-11-11，容器内 opencode + judge）

- **现象**：容器内 opencode 调本地 judge（vLLM，max_model_len=8192）报 `max_completion_tokens=32000 cannot be greater than max_model_len=8192`——配置里明明写了 `maxOutputTokens: 8192`。
- **根因**：opencode v2.2.5 的 config schema 中模型限制字段是 `"limit": {"context": N, "input": N, "output": N}`；`maxInputTokens/maxOutputTokens` 是 v1 风格字段，**被静默忽略不报错**。对网关模型无感（服务端不严格校验），对 max_model_len 严格的 vLLM 才暴露。
- **修复**：模型条目改用 `"limit": {"context": 8192, "output": 4096}`。判别法：看 `opencode.ai/config.json` schema（`$defs/Config.ModelEncoded/properties/limit`）。
- **附带发现**：opencode 官方 install 脚本（curl opencode.ai/install）在该容器内卡死无输出；直接从宿主机 npm 包拷原生二进制（`@opencode/cli/bin/opencode.exe`，aarch64 ELF，248M）到 `/usr/local/bin` 即可，openai-compatible provider **已内置二进制**，无需 node/npm、无需运行时联网（断网模拟验证通过）。

### E-8｜容器内 opencode 与宿主机 opencode 撞服务端口 49374（2026-11-11，deepeyes-v2）

- **现象**：`docker run --net=host` 的容器里跑 `opencode run` 报 `Managed service port 49374 on 127.0.0.1 is already in use`。
- **根因**：宿主机自己也跑着 opencode（默认服务端口 49374），host 网络模式下端口共享直接冲突。
- **修复**：容器内 `opencode service set port 49555`（配置落盘 `~/.config/opencode/service.json`，已烘焙进 v2 镜像）。离线镜像若再撞，同命令换端口。
- **附带**：judge 服务（142:18901）有 `max_new_tokens=64` 打分优化上限——做 opencode 问答连通性没问题，但 **agent 工具调用会静默无输出**（64 token 发不完 tool_call XML）。这不是 bug，是打分部署形态；离线 Qwen3.5-9B 服务（`/opt/scripts/serve_qwen35_9b.sh`）无此限制。

### E-7｜142 冒烟启动两类陷阱：docker exec -d 静默失败 + batch 整除断言（2026-11-11，142 迁移）

- **坑 1：`docker exec -d` 跑长启动命令偶发静默失败**——命令"发出去了"但容器内什么都没起（日志文件都不创建），138/142 都复现过。**修复**：改写成 `docker exec -d <c> bash -c '... nohup ... > log 2>&1 &'` 形式，或落盘成脚本再 exec；启动后必须回查日志文件/进程存在。
- **坑 2：`AssertionError: real_train_batch_size (64) must be divisible by minimal possible batch size (N)`**——batch 64（8×8）必须被 `NDEVICES_PER_NODE` 整除：6 卡 ❌、4 卡 ✅、8 卡 ✅。142 与 judge 共存时用 4 卡（0-3）冒烟即可。
- **教训**：142 侧冒烟必须设 `NDEVICES_PER_NODE=4`（新脚本已支持 `${NDEVICES_PER_NODE:-8}` 覆盖）。

### E-6｜142 起独立 vLLM 服务报 "Alloc Stream resource failed: Too many streams are created"（2026-11-11，142 eval 服务）

- **现象**：142 新容器内起 eval 服务（eval_smoke_v1_142，TP=2）连续两次失败：第一次在 NPUGraph capture 29/35 时报 `rtStreamWaitEvent ... resource alloc fail`，第二次（修了 AOT + enforce-eager 后）仍在 Worker 初始化阶段报 `Alloc Stream resource failed. Reason: Too many streams are created`。
- **根因**：卡 0-3 刚跑完训练冒烟，进程退出后**驱动级资源没放干净**（每卡残留 ~3.4G 无主 HBM + 流表脏），新进程在这几张卡上分配流资源直接失败。与显存余量无关（57G 空闲照样挂）。
- **修复**：**换干净卡**（卡 4/5 从未被冒烟占用）秒通。脚本同时补了两个防护：`export VLLM_USE_AOT_COMPILE=0`（E-4 同源，ray 外裸起服务也必须关）+ `--enforce-eager`（judge 同款，绕开 capture 路径）。已改进 `serve_eval_model.sh`。
- **判别法**：`npu-smi info` 里某卡有占用但 Process 表查不到对应进程 = 驱动级残留，该卡暂视为"脏卡"绕开；容器整体重启/服务全停重启后可复用。
- **注意**：恢复 judge TP=8 时会用回卡 0-3，若同样报 Too many streams，先 docker restart deepeyes-v1 再起（需重新评估打分窗）。

### E-5｜推理评测 400 "auto tool choice" + 工具调用解析不到（2026-11-10，推理链路验证）

- **现象①**：`/v1/chat/completions` 带 `tools` 参数即 400：`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`。
- **修复①**：serve_eval_model.sh 加 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`（Qwen3.5 的 XML 属性式 `<tool_call><function=x><parameter=y>` 格式对应 qwen3_xml parser；该 vllm 版本 tool_parsers 目录下确认存在）。
- **现象②**：开了 parser 后 eval 脚本仍解析不到工具调用——server 端 parser 把调用解析进了 OpenAI 响应的 `message.tool_calls` 字段，content 里不再有原始 `<tool_call>` XML，eval 脚本只对 content 做 regex 所以扑空。
- **修复②**：eval_agent.py 优先读 `message.tool_calls[].function`（name/arguments JSON），regex 作为裸文本兜底。
- **现象③**：评测 acc=0——eval 脚本自造的 system prompt + 裸问题与训练 prompt 不一致，模型失去格式约束。
- **修复③**：复刻训练构造（CustomRLHFDataset）：用 parquet 的 system 消息（含 `<tools>` schema）+ user 消息全文（含 `Think first... Format strictly as: <think>...` 指令），并同样做 `image_zoom_in_tool→image_zoom_in_rel_tool` 名称替换。
- **现象④**：基座模型最终答案不包 `<answer>` 标签 → 训练侧 compute_score 本就有 fallback（无标签记 format error 并取 `</think>` 后文本判分），eval 侧同步补齐 fallback，acc 恢复正常（8/8）。
- **经验**：**评测脚本的 prompt 构造必须与训练 dataset 完全一致**（同 system、同格式指令、同工具名替换），否则 eval 结果失真；serve 重启前确认旧 EngineCore 进程杀干净（HBM 释放），否则报 "Free memory 9.63/60.96 GiB"。

### E-4｜vLLM 引擎 aot_compile 崩溃（torch 2.10 默认开启 AOT）（2026-11-10，deepeyes-v1 冒烟）

- **现象**：训练 rollout 阶段 vLLM EngineCore 报 `RuntimeError: aot_compile is not supported by the current configuration. Please make sure torch.compile is enabled with the latest version of PyTorch (current using torch: 2.10.0+cpu)`，训练必挂。
- **根因**：vLLM 0.27.1 的 `envs.py: use_aot_compile()` 在 torch≥2.10 时**默认开启** AOT 编译路径（`VLLM_USE_AOT_COMPILE` 默认 1），而 vllm-ascend nightly (0.27.1rc1) 的 ACL graph 捕获链路走不通该路径。旧容器当年能跑，靠的是 ray daemon 环境或缓存差异，不可复现也无谓深挖。
- **修复**：在 `run_full_v1.sh` 传 `+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_AOT_COMPILE='"0"'`。**注意 hydra 陷阱：值必须写成 `'\"0\"'`（引号内引号传字符串）**，写 `=0` 会被解析成 int，ray 报 `TypeError: str(dict) expected`。
- **验证**：改后 Capturing CUDA graphs (PIECEWISE) 35/35 通过，训练/推理均正常。

### E-3｜容器内 localhost 无法解析 → Ray 连接失败（2026-11-10，deepeyes-v1）

- **现象**：新容器内 `socket.gethostbyname("localhost")` 报 `gaierror: Name or service not known`，Ray 初始化 `Could not find any running Ray instance`。
- **根因**：宿主机 `/etc/hosts` 本身没有 `127.0.0.1 localhost` 条目（机器以 `71.10.29.138 syn-138` 裸奔），docker 默认拷贝宿主 hosts，容器内 glibc 仅靠 `files` 找不到 localhost。
- **修复**：`docker run` 加 `--add-host localhost:127.0.0.1 --add-host localhost.localdomain:127.0.0.1`。
- **经验**：重建任何该机器上的容器都要带这两个 add-host（已写进 RUNBOOK §2.4 备忘）。

### E-2｜新容器无法访问 NPU：旧容器"按命名空间独占"设备（2026-11-10）

- **现象**：用 `deepeyes-train-infer:v1` 在 138 本机按 CASE §3 重建容器（`deepeyes-v1`），import 链全部正常，但容器内 `npu-smi info` 报 `dcmi model initialized failed, because the device is used. ret is -8020`，`torch.npu.device_count()=0`。宿主机和旧容器（deepeyes）的 npu-smi 一切正常。
- **排查过程**（逐项排除，最后 dmesg 定案）：设备节点/挂载/HostConfig/env 全部与旧容器一致；基础镜像同样复现（排除新镜像问题）；依次排除 seccomp、cap-add=ALL、SELinux label、cgroup device rule、--pid=host（均无效）；**只有 --privileged 能绕过**。
- **根因（dmesg 实锤）**：`[ascend][uda][ERROR] uda_occupy_dev_by_ns: Conflict open udevid (udevid=0; access_ns=a9cad8f3; ns=<新容器ns>)` ——昇腾驱动 UDA 的**按命名空间设备占用机制**：死训练所在旧容器先打开了设备，8 卡被其命名空间（`00000000a9cad8f3`）独占；任何新容器（=新 ns）open 设备即冲突（errno 87=EUSERS → dcmi -8020 "device is used"）。
- **结论**：旧训练不停，一切新容器都碰不了 NPU（推理/训练/评测全阻塞）；`docker stop deepeyes` 释放占用后新容器立即可用。宿主机 npu-smi 正常是因为宿主 ns 是占用机制的豁免方/管理者。
- **经验**：① 以后在"有训练在跑"的机器上起新 NPU 容器报 -8020，先查 `dmesg | grep uda_occupy`，不要浪费时间去 bisect 容器参数；② privileged 能绕过说明该检查对特权进程豁免——但**不要**用 privileged 跑训练来绕（两套容器同时操作设备有踩踏风险，且占用语义未验证）；③ 这也是"训练占卡不释放"的另一种表现——E-1 死训练的代价又多了一条。

### E-1｜正式训练 run 在 step ~183 发生 policy collapse，此后 100+ 步零梯度空转（2026-11-10 发现）

- **现象**：`full_outer_138_v3.log`（run 于 11-04 启动）——
  - step 165-170：`actor/grad_norm` 从 1.2 飙至 **15.0**（正常 ~0.4-1.2），`ppo_kl` 从 0.002 爬到 0.0089；
  - step 178-182：score 跌至 0.18 → -0.17，ppo_kl 冲到 **0.057**（严重偏离）；
  - **step 184 起**：`critic/score/mean` 恒等于 **-0.20000003**（=0.8×0+0.2×(-1)+1.2×0：无答案、判 0 分、无工具），`pg_loss=0`、`grad_norm=0`、`advantages≡0`——所有轨迹奖励完全相同 → 组内优势为零 → **无任何梯度，训练死亡但进程不报错继续跑**；
  - 佐证：step 240 val 的 `vstar acc=0.0`、`tool_used=0.0`、`answer_length≈38000`（正常 11~1305）；response_length/mean 崩溃后缩到 114。
- **发现方式**：为回答"怎么看指标图"写了日志解析脚本（`eval_prep/plot_train_metrics.py`），出图后悬崖一目了然。
- **处理（进行中）**：
  - 可恢复资产：崩溃前最后的健康点 **step 170 的 HF 导出已存在**：`models/eval_qwen35_9b_step170`（val acc 0.708，但已较 step 80 的 0.775 回落——崩溃前已有劣化迹象）；
  - ckpts/full 里最新 2 个 checkpoint（~285/290）全是崩溃后权重，无恢复价值；崩溃前 checkpoint 因 `max_actor_ckpt_to_keep=2` 已被覆盖删除；
  - judge 服务曾被重启（v1/models 的 created 时间戳从 11-08 变为 11-10），但崩溃根因是模型退化（tool_used/answer_length 不依赖 judge），judge 只是并发受害者/巧合。
- **根因（初步）**：KL 失控（kl_coef=0.0 无 KL 约束 + entropy 未监控）→ grad_norm 尖峰 → 输出分布塌缩到"短输出无答案"模式 → 零方差奖励 → GRPO 无梯度。触发点未定（可能是某步大梯度更新）。
- **预防（下次 run 必改）**：① LOGGER 加 tensorboard 便于盯曲线；② 监控 `actor/ppo_kl` 与 `actor/grad_norm`（超阈值告警）；③ 评估加小 kl_coef 或 clip 更紧的 pg 上限；④ 期待上游支持"连续 N 步 advantage 全零则 abort"。
- **关联**：ARCHTED_ISSUES.md I-9（向上游提"零方差奖励静默死亡"告警机制）。



> 登记模板（复制使用）：
>
> ### [日期] 简短标题
> - **现象**：报错原文 / 异常行为（贴关键日志行）
> - **上下文**：什么操作、什么配置、哪台机器
> - **定位过程**：怎么找到根因的（查了什么日志/文档/代码）
> - **根因**：
> - **处理**：改了什么（配置/代码/环境），是否需要同步给同事文档
> - **状态**：已解决 / 规避 / 未解决

### （暂无）

---

## C. verl 核心本地补丁登记（重要资产，丢失即重打）

> 同事在**训练机的 verl 核心检出**上打的 3 个补丁，均未提交到远程。拿到检出后立即备份并
> 提交到我自己的 fork；拿不到就按下面说明重打。

| # | 文件 | 补丁内容 | 来源依据 |
| --- | --- | --- | --- |
| 1 | `verl/workers/utils/padding.py` | 采用上游 #8413 的 `nested_tensor_from_jagged(..., jagged_dim=token)` | verl PR #8413 |
| 2 | `verl/utils/tensordict_utils.py::index_select_tensor_dict` | nested `unbind` 失败时按 `values`/`offsets` 还原样本 | 同事自研兜底 |
| 3 | `verl/experimental/agent_loop/tool_parser.py` | `ast.literal_eval` 结果为 `Ellipsis` 时保留原始字符串 `"..."` | 同事自研修复 |

- [x] 补丁已确认在位：`/home/h00516332/deepeyes_run/verl`（容器 `/workspace/verl`），2026-11-08 grep 验证通过
- [ ] 补丁已提交到我自己的仓库（备注 commit）——**待办：把 deepeyes_run/verl 纳入 git 管理并推送**

## D. 待办问题（继承了同事未完成项）

- [ ] judge prompt 超长 400：在 `_judge_accuracy` 截断 `answer_text`（保留尾部最终答案）或 judge `max_model_len` ≥16384（原问题 #12 副作用）
- [ ] 节点 HDK 升级 ≥25.5（同时解决问题 #8 ACL graph 与 #15 IPC）
- [ ] 训练后独立评测脚本（参考案例文档 §4.1 实现）
