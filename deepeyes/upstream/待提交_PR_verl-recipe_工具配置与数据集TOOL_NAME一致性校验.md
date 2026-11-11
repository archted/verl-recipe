# PR：[recipe/deepeyes] validate tool-config names against dataset TOOL_NAME at startup (fail fast on mismatch)

> **渠道**：verl-project/verl-recipe PR ｜ **状态**：待提交 ｜ **台账 ID**：I-15 / E-11
> **提交前检查**：需写校验代码（dataset 加载处 + agent loop 注册处各一处 assert/warn）；先提 I-10 issue
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

**PR title:** `[recipe/deepeyes] fail fast when tool_config_path names don't match the dataset's TOOL_NAME`

```markdown
## Change
At dataset init, compare the tools declared in `tool_config_path` with the dataset's `TOOL_NAME`; on mismatch,
log a loud ERROR listing both sides (we lost a full training round to a silent mismatch: every tool call failed,
tool_used stayed 0, tool reward contributed 0 — training ran to completion with no warning).

## Validation
Synthetic mismatch → startup fails with actionable message; matching config → unchanged behavior.
```
