# verl-recipe (deepeyes): default tool config uses absolute-coordinate tool while Qwen3.5 emits 0-1000 relative coords; tool name mismatch fails silently

> **渠道**：verl-project/verl-recipe issue（关联同事 open 的 #166/#167，先评论后开） ｜ **状态**：可提交 ｜ **台账 ID**：I-15 / E-11 / P0-1
> **提交前检查**：在 #167 下先评论确认维护者意见，避免重复开题
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug (two related defaults in `recipe/deepeyes` for Qwen3.5)

1. **Coordinate system mismatch.** Qwen3.5 emits bounding boxes in **0-1000 relative coordinates**, but the recipe's default `tool_config_path` / `multi_turn.tool_config_path` points to `image_zoom_in_tool_config.yaml` (the **absolute-pixel** tool). The model's tool calls then use relative numbers against an absolute-coordinate cropper — crops are wrong/degenerate, yet nothing errors.
2. **Silent failure on tool-name mismatch.** If `tool_config_path` names a tool that does not match the dataset's `TOOL_NAME` (we hit exactly this: config declared one tool name, the parquet's `extra_info` carried another), **every tool call fails with no warning at all**: training runs to completion, `tool_used` stays 0, the tool reward channel silently contributes 0. We lost a full training round to this before a case-study audit caught it.

### Evidence
- Our audit + fix: switched defaults to `image_zoom_in_rel_tool_config.yaml` (0-1000 relative tool) in both the yaml (line 36) and the launcher script (line 165); training then shows non-zero tool reward from the first steps (score/max reaching the 2.0 cap implies the tool channel works).
- The silent-mismatch path: no validation exists between `tool_config_path`'s declared tool name and the dataset's `TOOL_NAME` / the tool schema registered in the agent loop.

### Suggested fix
1. Default Qwen3.5 configs to the **relative-coordinate** tool (`image_zoom_in_rel_tool_config.yaml`).
2. At dataset/agent-loop init, validate that every configured tool name matches the dataset's `TOOL_NAME` and fail fast (or at least WARN with counts) on mismatch.
3. Document the absolute-vs-relative coordinate requirement per model family in the deepeyes README.
