# DTS 单草稿（内部提交用，archted 手动提交）

> 提交渠道：内部 DTS 系统（GitHub 上提不了内部文档/环境问题）
> 本文件由 arched wiki 维护，提交后把单号回填到 ARCHED_ISSUES.md §5

---

## DTS 单 1（本单）：deepeyes 配方 REF_MODEL_PATH 死占位符（与开源 issue/PR 联动）

**单标题：** DeepEyes recipe 启动脚本 REF_MODEL_PATH 为死占位符且语义误导（影响开源仓提交流程）

**问题类别：** 文档/工具缺陷

**现象描述：**
verl-recipe 仓 `deepeyes/run_deepeyes_grpo.sh`（源自 verl 仓 recipe 迁移，迁移 PR #7）第 16 行
`REF_MODEL_PATH=ref_model_path` 为字面量占位符，从未被赋真实值，第 30 行作为
`actor_rollout_ref.model.path` 传入。照原样运行必然因模型路径不存在而启动失败。
且变量名误导：`actor_rollout_ref.model.path` 实为 actor/rollout 主模型路径；
该配方已关闭 KL（kl_coef=0.0、use_kl_loss=False），全程不加载 ref 模型，
按名字填"参考模型"路径会静默训错模型。

**影响：** 新接手同学按脚本直接跑必踩坑；误导性命名可能造成静默错误（训错模型）。

**修复建议：** 改为 `MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}` 带守卫默认值
+ 注释说明该参数是 actor 主模型路径、本配方不加载 ref。

**已提交开源侧：**
- verl-project/verl-recipe issue：<提交后回填编号/链接>
- verl-project/verl-recipe PR（fork archted/verl-recipe 分支 fix-deepeyes-run-script-model-path）：<签 CLA 后提交，回填编号>

**关联：** ARCHED_ISSUES.md I-8

---

## DTS 单 2（备稿，可一并提）：同事交接文档中 judge 部署信息缺失/误导（I-3）

**单标题：** DeepEyes 交接文档 judge 部署信息与实际不符，重建环境时无法照做

**问题类别：** 文档纠错

**现象描述：**
1. 交接 TROUBLESHOOTING「环境」表未记录 judge 服务真实配置（实测：模型
   Qwen3-30B-A3B-Instruct-2507、TP=8 占满 142 全部 8 卡、max_model_len 8192、
   override-generation-config max_new_tokens=64、端口 18901、served-model-name judge）；
2. RUNBOOK §6.1 示例写 judge 模型为 Qwen2.5-32B，与实际不符；
3. 「单步耗时分析」记录 79-92 分钟/步为 142 节点语境，当前 138 节点（HDK 25.5.2 +
   bypass_mode）实测约 29 分钟/步，差异未标注，易误判训练异常。

**影响：** judge 服务故障时无法按文档重建；耗时差异会造成误判。

**修复建议：** 在交接文档补录 judge 真实部署命令（从 142 容器 ps 还原）与节点语境标注。

**关联：** ARCHED_ISSUES.md I-3
