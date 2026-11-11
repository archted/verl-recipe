# PR：[recipe/deepeyes] don't force TORCHDYNAMO_DISABLE=1 when use_torch_compile is enabled

> **渠道**：verl-project/verl-recipe PR ｜ **状态**：可提交 ｜ **台账 ID**：I-17
> **提交前检查**：一行修复 + 启动告警；先提 issue 查重
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

**PR title:** `[recipe/deepeyes] respect use_torch_compile: stop force-exporting TORCHDYNAMO_DISABLE=1`

```markdown
## Problem
`run_deepeyes_grpo_qwen3_5.sh` line 64 unconditionally exports `TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}`,
silently defeating `actor.use_torch_compile=True` passed by launchers. Measured impact on our 9B NPU run:
MFU 1.6%, AICore ~30% util, update_actor 18.3 min of a 24-26 min step — eager dispatch dominates, and the
user has no way to see that compile is off.

## Change
- Gate the export: only disable Dynamo when torch compile is not requested.
- Add a startup WARNING when both `use_torch_compile=True` and `TORCHDYNAMO_DISABLE=1` are in effect.
```

**代码位置：** 待打分支（改动 < 10 行；配方仓 scripts 域）。
