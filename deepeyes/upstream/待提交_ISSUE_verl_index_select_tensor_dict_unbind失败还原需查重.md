# verl: tensordict index_select_tensor_dict fails when unbind returns Ellipsis-containing leaves (needs dedup + minimal repro)

> **渠道**：verl-project/verl issue ｜ **状态**：待提交 ｜ **台账 ID**：I-19（本地补丁②）
> **提交前检查**：先查重（同事 #8550/#8559 只覆盖 tool_parser 侧）；准备 5 行最小复现脚本
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
`verl/utils/tensordict_utils.py::index_select_tensor_dict` breaks when a jagged nested tensor's unbind
produces leaves containing `...`/Ellipsis sentinels (observed with `nested_tensor_from_jagged` outputs),
failing downstream JSON serialization.

### Local patch we carry
unbind-failure fallback path that rebuilds from values/offsets (patch ② in our local set; patch ① tracks
upstream PR #8413, patch ③ is #8560).

### To collect before submitting
- [ ] upstream dedup search results
- [ ] 5-line minimal repro (tensor shapes + failing call)
- [ ] confirm it still reproduces on latest main
