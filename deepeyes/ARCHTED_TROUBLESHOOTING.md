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
