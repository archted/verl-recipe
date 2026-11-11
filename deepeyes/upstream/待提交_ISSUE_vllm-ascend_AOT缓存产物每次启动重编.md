# vllm-ascend: AOT/inductor cache artifacts fail to load on Ascend; every worker recompiles 1-2 min at startup

> **渠道**：vllm-project/vllm-ascend issue ｜ **状态**：待提交 ｜ **台账 ID**：I-5 / T-#18
> **提交前检查**：需补完整日志片段（每次启动的 WARNING 块 + 时间戳）+ 确认缓存目录权限/路径假设
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
AOT compile cache artifacts (including inductor Triton wrappers) generated on the first startup cannot be
loaded on Ascend on subsequent starts; every engine worker recompiles 1-2 minutes per start and floods the
log with load-failure WARNINGs.

### To collect before submitting
- [ ] one full WARNING block with paths it tried to load
- [ ] timestamps showing the per-worker recompile cost
- [ ] confirmation the cache dir is writable and stable across restarts
