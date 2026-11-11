# PR：[recipe/deepeyes] gate tool reward on the tool-success marker (restore anti-hacking check)

> **渠道**：verl-project/verl-recipe PR ｜ **状态**：待提交 ｜ **台账 ID**：I-2
> **提交前检查**：先提 I-2 issue；从 outputs/ 抽一条“工具执行失败仍渲染 <tool_response>”的真实轨迹附上；签 CLA
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

**PR title:** `[recipe/deepeyes] require tool-success marker before granting tool reward`

```markdown
## Change
- `has_tool_usage` additionally requires the success marker text
  ("Zoomed in on the image to the region (relative:") in the trajectory, matching the case-study
  implementation's anti-reward-hacking check that was lost in the Qwen3.5 adaptation.
- Emit the failed-tool-call ratio as a diagnostic metric.

## Evidence
(attach one real trajectory where the tool execution failed yet the sample received full tool reward)
```

**代码位置：** 待打分支（修法已在本地配方验证过 TOOL_SUCCESS_MARKER 门控）。
