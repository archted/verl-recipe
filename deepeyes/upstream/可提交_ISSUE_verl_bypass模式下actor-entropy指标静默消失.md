# verl: actor/entropy metric silently disappears when rollout_correction.bypass_mode=True

> **渠道**：verl-project/verl issue ｜ **状态**：可提交 ｜ **台账 ID**：I-4
> **提交前检查**：先查重；附 1 行代码定位即可提
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### System Info
- verl 0.10.0.dev0, config: `algorithm.rollout_correction.bypass_mode=True`

### Describe the bug
With `algorithm.rollout_correction.bypass_mode=True`, the `actor/entropy` metric stops being produced entirely — silently. Our monitoring relies on entropy as an early-warning signal for policy collapse (we hit a real GRPO collapse where entropy would have been the first visible symptom, see related issue about zero-variance rewards).

Observed: with bypass_mode=False the metric appears every step; with bypass_mode=True it is absent from metrics with no warning. Code-wise, entropy calculation is skipped when the bypass path is taken (entropy is only computed along the non-bypass path).

### Expected behavior
Either keep computing entropy (it is cheap relative to training), or produce a distinct metric (e.g. `actor/entropy_proxy`) / log an explicit INFO that entropy observability is disabled under bypass mode.

### Why it matters
bypass_mode is a performance optimization; silently removing a key observability metric makes the optimization harder to adopt safely. In our collapsed run, bypass_mode=True + no entropy + no zero-variance detection meant zero early-warning signals.
