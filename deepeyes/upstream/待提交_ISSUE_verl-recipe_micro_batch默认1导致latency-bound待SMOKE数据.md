# verl-recipe (deepeyes): default ppo_micro_batch_size_per_gpu=1 makes actor update latency-bound (MFU 1.6%)

> **渠道**：verl-project/verl-recipe issue / #167 评论 ｜ **状态**：待提交 ｜ **台账 ID**：I-20
> **提交前检查**：等 micro 1→8 SMOKE 实测数据（update_actor/MFU/reserved 三指标）回来后提，附前后对比
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug (draft)
Qwen3.5-9B GRPO run on 8×910B3: `ppo_micro_batch_size_per_gpu=1` + full offload → actor update takes 18-19 min
of a 24-26 min step; verl-reported MFU 1.6%, AICore util averages ~30% (peaks 100%, valleys ~1%), HBM 40/64 GB,
host ~88% idle. This is dispatch/latency-bound eager execution, not a compute or memory limit: kernel-launch
count scales with micro-batch count, not tokens.

Note: the case-study implementation for the same model family uses micro_batch=8 — the recipe default of 1 is
the deviation.

### Pending
- [ ] SMOKE measurement with micro 1→8 (timing_s/update_actor, perf/mfu/actor, max_memory_reserved_gb)
- [ ] before/after numbers to make the issue concrete
