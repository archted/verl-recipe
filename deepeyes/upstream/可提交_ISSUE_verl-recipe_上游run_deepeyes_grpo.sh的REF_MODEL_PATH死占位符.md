# verl-recipe (deepeyes): run_deepeyes_grpo.sh fails out of the box — dead REF_MODEL_PATH placeholder passed as actor model path

> **渠道**：verl-project/verl-recipe issue ｜ **状态**：可提交 ｜ **台账 ID**：I-8
> **提交前检查**：英文全文已有：deepeyes/drafts/I8_issue_verl-recipe.md（查重已做：上游无人提过，#4989 不撞车）；PR 分支已备 bd82d24
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

（正文见 `deepeyes/drafts/I8_issue_verl-recipe.md`，英文完整版，直接复制粘贴。要点：line 16 `REF_MODEL_PATH=ref_model_path` 字面量占位符被 line 30 用作 `actor_rollout_ref.model.path`，开箱必崩；且该名字有误导——verl 中该字段配置的是 actor/rollout 基座模型而非 ref policy；上游 main 至今未修。）
