# verl：[GRPO] Silent training death: zero-variance rewards produce zero gradients with no warning for 100+ steps

> **渠道**：verl-project/verl issue（代码可另上 PR：连续 N 步 advantage 全零熔断） ｜ **状态**：可提交 ｜ **台账 ID**：I-9 / E-1 / DTS-001
> **提交前检查**：先查重（搜 zero variance / all-zero advantage）；签 CLA 后再提配套 PR
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### System Info
- verl 0.10.0.dev0 (recipe/deepeyes GRPO), vllm-ascend 0.27.1rc1, vLLM 0.27.1, CANN 9.1.0, torch_npu 2.10.0.post4
- 8×Ascend 910B3, single node, GRPO (rollout.n=8, train_batch_size=64)

### Describe the bug
A production GRPO run suffered a **policy collapse at step ~183 and then trained "silently" for 100+ steps with literally zero gradient**, burning ~50 GPU-hours × 8 NPU cards, with **no warning, no error, and no abort**.

Evidence from the real run (`full_outer_138_v3.log`):
- step 165-170: `actor/grad_norm` spikes 1.2 → **15.0** (normal 0.4-1.2); `actor/ppo_kl` climbs 0.002 → 0.0089
- step 178-183: `critic/score/mean` falls 0.18 → -0.17; ppo_kl hits **0.057**
- **step 184 onwards**: `critic/score/mean` is **exactly -0.20000003** for every sample (= 0.8×0 + 0.2×(-1) + 1.2×0: no answer, judge-scored 0, no tool), hence `critic/advantages ≡ 0`, `actor/pg_loss=0`, `actor/grad_norm=0` — every trajectory gets the same reward, group-relative advantage is identically zero, so **nothing is learned but the loop keeps running**
- step 240 validation: `val-core/vstar/acc/mean@1=0.0`, `tool_used=0.0`, `answer_length≈38000` (normal 11~1305) — the policy has collapsed to a degenerate mode
- step 184→292 = **108 steps (~50 h × 8 cards) of pure compute waste**

### Why this is a framework-level gap
"Zero-variance reward → zero advantage → zero gradient" is a known GRPO failure mode, but verl neither detects nor reports it. In our case the only symptom in the log was `score/mean` being constant; a user watching loss curves sees nothing (loss is exactly 0, which looks like "perfectly converged").

### Suggested fix
1. Add a circuit breaker: after N consecutive steps (configurable, e.g. `VERL_ZERO_VAR_ABORT_STEPS=5`) where `critic/advantages` are all-zero (or reward is constant across the whole batch), log an explicit ERROR and **abort training / roll back to the last checkpoint**.
2. Emit a warning the first time reward variance is 0 for a single step (cheap check, catches hacking/collapse early).
3. Bonus: when `bypass_mode=True`, `actor/entropy` is no longer produced (separate issue), which removes the other early-warning signal exactly when it matters most.

We are happy to contribute a PR implementing the breaker (proposed design: env var `VERL_ZERO_VAR_ABORT_STEPS=N`; N consecutive steps with all-zero advantages → explicit ERROR log + abort + auto-rollback to the last checkpoint), pending maintainer feedback on the preferred UX (hard abort vs. warn-only with a metric).

### Impact
Any long agentic-RL run can hit this; without detection the failure costs are measured in days of idle cluster time.
