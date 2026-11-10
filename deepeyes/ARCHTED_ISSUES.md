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
| **I-1** | 代码缺陷+改进 | **judge prompt 超长 400 → acc 记 0**：`deepeyes.py::_judge_accuracy` 对 question/answer_text 无截断；judge `max_new_tokens=64` 副作用下，answer 过长时 prompt 超 8192 被 vLLM 拒（400），重试 3 次全败记 0 分（实测 ~8 次/步污染奖励）。顺带：4xx 是确定性失败却仍重试 3 次（sleep 1s+2s）浪费时延 | T-#12 全记录 + 现跑日志仍每步出现 400 + 代码确认无截断（deepeyes.py:295-344） | 已验证 → **可提 PR**（修法：按 token 预算截断 answer_text 保留尾部最终答案 + question 截断 + 4xx 不重试） | verl PR + DTS |
| **I-2** | 代码缺陷（回归） | **tool_reward 计入失败的工具调用（丢了防 reward-hacking 检查）**：`compute_score` 的 `has_tool_usage` 只查 `<tool_call>`/`<tool_response>` 标签存在（deepeyes.py:261-265）。bbox 非法时工具执行失败，但错误响应同样渲染为 `<tool_response>` → 仍得满分工具奖励。案例团队（同是 Qwen3.5）为此专门加过 `"Zoomed in on the image to the region (relative:" in solution_str` 成功标记检查（案例文档 §2.4 明确记录 reward hacking 现象），同事版本丢失该防线 | 代码 grep 确认无 "Zoomed in" 检查 + 案例文档 §2.4 记录 | 已验证 → **可提 PR**（修法：tool_used 要求含成功标记文本；补充：失败调用占比可作诊断指标）。提交前补一条实轨迹证据（outputs/ 里找 error 工具响应渲染样例） | verl-recipe PR（关联 #166/#167）+ DTS |
| **I-3** | 内部文档错误 | 同事 TROUBLESHOOTING「环境」表未记录 judge 真实模型/部署形态（实际 Qwen3-30B-A3B-Instruct-2507、TP=8 占满 142 全部 8 卡、max_model_len 8192）；RUNBOOK §6.1 示例写 Qwen2.5-32B 会误导重建；「单步耗时分析」的 79-92 分钟是 142 语境，与 138 实测 29 分钟差异未标注 | 142 实测 + 本机 run_full_138.sh | 已验证 | DTS / 直接在 archted 仓 wiki 修正 |
| **I-4** | 监控/文档 | `algorithm.rollout_correction.bypass_mode=True` 后 `actor/entropy` 指标不再产出（TROUBLESHOOTING 提速实验实测），监控面板/告警若依赖该指标会误判 | TROUBLESHOOTING「提速实验」+ 代码（calculate_entropy=False 时不计算） | 候选（需确认上游是否已知/有 issue） | verl issue（先查重）或文档 PR |
| **I-5** | 推理引擎 | vllm-ascend AOT compile 缓存产物（含 inductor Triton wrapper）在 Ascend 无法加载，每次启动每 worker 重编 1-2 分钟并刷 WARNING（T-#18） | TROUBLESHOOTING #18 | 候选 → 需收集完整日志片段 | **vllm-ascend issue（蓝海，同事零提交）** |
| **I-6** | 工程健壮性 | 同事 `#8555/#8553` 被关未合并后，配方侧依赖变通（绝对路径 + prefilter），若他人按 README 直接跑会踩同样坑——可在 #167 下评论提醒或在 README 补充说明 | PR 关闭记录 | 候选 | verl-recipe（评论/文档 PR） |
| **I-9** | 训练健壮性（高价值） | **GRPO 零方差奖励 → 训练静默死亡，无告警无中止**：正式 run 实测（E-1）——step 184 起 `critic/score/mean` 恒 -0.2、`advantages≡0`、`pg_loss=0`、`grad_norm=0`，训练循环不报错继续跑了 100+ 步（~50 小时 × 8 卡白白空转，ETA 还剩 109h）。val 的 acc=0/tool_used=0 佐证模型输出已塌缩。verl 应支持：连续 N 步 advantage 全零（或 score 恒等于某常数）时告警/中止/回滚到上一 ckpt。附带：bypass_mode 下 entropy 不产出（I-4）加剧了无监控 | 日志 `full_outer_138_v3.log` step 184-291 + `eval_prep/metrics_report/1_score.png` 悬崖图 | 已验证 → **可提 issue（强证据链）**，PR 可做"全零 advantage N 步熔断" | verl issue + DTS |
| **I-8** | 上游配方脚本缺陷 | **verl-recipe 上游 `recipe/deepeyes/run_deepeyes_grpo.sh` 的 `REF_MODEL_PATH` 是死占位符**：第 16 行 `REF_MODEL_PATH=ref_model_path`（字面量，从未赋真实值），第 30 行用作 `actor_rollout_ref.model.path=${REF_MODEL_PATH}` → 照原样运行必然报"路径不存在"。上游 commit 21892b9（migrate recipes from verl #7）带入，非同事改动。修法：改为 `REF_MODEL_PATH=${REF_MODEL_PATH:?set to your base model dir}` 或在 README 标注必填 | `git log -- deepeyes/run_deepeyes_grpo.sh` + 第 16/30 行 | 已验证 → 小 PR（低优先级，但稳定可收） | verl-recipe issue/PR |
| **I-7** | 待挖掘 | deepeyes.py 其余逻辑（answer 抽取回退链、1000 字符惩罚）、tools/image_zoom_in_tool.py、run_*.sh、prefilter 脚本的逐行 review；`run_deepeyes_grpo_qwen3_5.sh` 的 PYTHON 解析器问题（T-#1 相关）是否已修 | 待做 | 待办 | — |

## 5. 提交台账（我提交的记录）

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
- [ ] I-1 写 issue + PR（等训练空档在 138 验证截断逻辑）
- [ ] I-2 从 outputs 抽一条"工具执行失败仍渲染 <tool_response>"的真实轨迹作证据，然后提 PR
- [ ] I-4/I-5 在上游查重后决定是否提
- [ ] 每周复盘新增候选
