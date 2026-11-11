# verl-recipe (deepeyes): run script exports TORCHDYNAMO_DISABLE=1 while config sets use_torch_compile=True — compile silently disabled

> **渠道**：verl-project/verl-recipe issue ｜ **状态**：可提交 ｜ **台账 ID**：I-17
> **提交前检查**：查重；确认上游脚本行号
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
`recipe/deepeyes/run_deepeyes_grpo_qwen3_5.sh` line 64:

```bash
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
```

while the deepeyes Qwen3.5 config path is normally launched with `actor_rollout_ref.actor.use_torch_compile=True`. The env var wins: Dynamo is hard-disabled process-wide and **the training runs eager while the user believes torch.compile is on**. No warning is printed anywhere.

### Evidence
- Both lines coexist in our launch pipeline (recipe script exports the env var; launcher passes `use_torch_compile=True`).
- Log shows the `+ export TORCHDYNAMO_DISABLE=1` line and zero dynamo/inductor output (no "Recompiling", no cache logs) over 41 steps.
- Perf context: measured MFU 1.6% with AICore util ~30% — dispatch/latency-bound eager execution is consistent with compile being off; per-token actor time 1.2 ms.

### Suggested fix
1. Remove the export, or gate it: only set `TORCHDYNAMO_DISABLE=1` when `use_torch_compile` is not True.
2. At startup, if `use_torch_compile=True` but `TORCHDYNAMO_DISABLE=1`, print a loud WARNING (this contradiction is invisible otherwise).

### Impact
Users silently lose the compile speedup; on NPU dispatch-bound stacks this can be the difference between ~26 min/step and single-digit minutes/step for a 9B model.
