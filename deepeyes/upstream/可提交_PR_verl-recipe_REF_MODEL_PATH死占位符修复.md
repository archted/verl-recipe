# PR：[recipe/deepeyes] fail fast with a clear message when REF_MODEL_PATH is not set

> **渠道**：verl-project/verl-recipe PR ｜ **状态**：可提交 ｜ **台账 ID**：I-8
> **提交前检查**：PR 分支已备：fork archted/verl-recipe 分支 fix-deepeyes-run-script-model-path（commit bd82d24，bash -n 已验）；预填 compare 链接在 deepeyes/drafts/I8_pr_create_url.txt
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

**PR title:** `[recipe/deepeyes] make REF_MODEL_PATH required and rename it to MODEL_PATH`

**正文要点（英文完整版见 drafts/I8_issue_verl-recipe.md 同目录说明）：**

```markdown
## Problem
`deepeyes/run_deepeyes_grpo.sh` line 16 sets `REF_MODEL_PATH=ref_model_path` — a literal placeholder that
nothing tells the user to replace — and line 30 passes it to `actor_rollout_ref.model.path`, so the script
crashes at startup with "path does not exist". The name is also misleading: in verl this field configures
the actor/rollout base model, not a reference policy (this recipe sets kl_coef=0.0, no ref weights are loaded).

## Change
- `REF_MODEL_PATH=${REF_MODEL_PATH:?set REF_MODEL_PATH to your base model dir}`
- (optional) rename to `MODEL_PATH` to match semantics.

## Validation
`bash -n` passes; script now fails immediately with an actionable message instead of a hydra model-load error.
```

**创建方式：** 细粒度 token 无官方仓 PR 写权限（API 403），用 `drafts/I8_pr_create_url.txt` 的预填 compare 链接手动创建。
