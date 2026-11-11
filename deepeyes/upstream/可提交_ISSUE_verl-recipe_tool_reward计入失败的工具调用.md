# verl-recipe (deepeyes): tool_reward granted for FAILED tool calls (regression: success-marker check lost)

> **渠道**：verl-project/verl-recipe issue ｜ **状态**：可提交 ｜ **台账 ID**：I-2 / DTS-003
> **提交前检查**：提交前从 outputs 抽一条真实“工具执行失败仍渲染 <tool_response>”轨迹作证据
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
In `deepeyes.py::compute_score`, the tool-reward gate (`has_tool_usage`) only checks that the trajectory **contains** `<tool_call>`/`<tool_response>` tags. When the model emits an invalid bbox, the tool execution **fails**, but the error message is still rendered as a `<tool_response>` — so the sample receives the full tool reward (weight 1.2, the largest term in `0.8×acc + 0.2×format + 1.2×tool`).

This is a regression against the original design: the case-study implementation for the same Qwen3.5 stack explicitly checks for the success marker text (`"Zoomed in on the image to the region (relative:"`) before granting tool reward, and its §2.4 documents the exact reward-hacking behavior this prevents (models spamming tool calls without correct usage). The Qwen3.5 adaptation in verl-recipe dropped that check.

### Why it matters
- Reward hacking: the cheapest way to farm 1.2 points is to emit syntactically valid tool calls, correct or not.
- We observed exactly this family of behavior in a collapsed run (tool_used stays high while accuracy goes to 0).

### Suggested fix
Gate `tool_used` on the success marker (as the case-study does), e.g. require `"Zoomed in on the image to the region (relative:"` in the trajectory; additionally log the failed-call ratio as a diagnostic metric. PR ready upon attaching one real failed-call trajectory as evidence.
