# verl: rolling checkpoint deletion (max_actor_ckpt_to_keep) destroys the last healthy checkpoint after a collapse

> **渠道**：verl-project/verl issue ｜ **状态**：可提交 ｜ **台账 ID**：I-22
> **提交前检查**：先查重（best-k / keep-healthy 相关讨论）
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
When a GRPO run collapses mid-training (policy collapse, reward goes constant), the rolling checkpoint policy `max_actor_ckpt_to_keep=2` keeps **overwriting** checkpoints with collapsed weights. By the time the collapse is noticed (which can be 100+ steps later if there is no zero-variance abort — see our related issue), **every retained checkpoint is already from the collapsed region**. The last healthy checkpoint (step 170 in our case, before a step-184 collapse) was deleted; we lost the ability to resume from a sane state and had to restart training from scratch.

### Suggested fix (any one of)
1. `max_actor_ckpt_to_keep` deletion should be suspended while a configured anomaly signal is active (e.g. zero-variance breaker counts > 0).
2. Add a "keep best-k by validation metric" option (`save_best_by=val-core/...-acc`) alongside the rolling window.
3. At minimum, document loudly that `max_actor_ckpt_to_keep` is unsafe for RL runs without automated collapse detection.

### Impact
Checkpoint deletion is irreversible; combined with silent collapse modes it converts a recoverable failure into a full restart (~9 days of single-node training in our setup).
