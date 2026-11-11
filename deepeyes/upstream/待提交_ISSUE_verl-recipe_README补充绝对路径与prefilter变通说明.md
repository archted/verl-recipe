# verl-recipe (deepeyes): README/workflow gaps bite new users after #8553/#8555 were closed unmerged

> **渠道**：verl-project/verl-recipe（文档 PR 或在 #167 评论） ｜ **状态**：待提交 ｜ **台账 ID**：I-6
> **提交前检查**：决定走评论还是文档 PR；稿子基于本仓已实测的两条变通（绝对路径 tool_config + prefilter）
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Draft point
Two workarounds are load-bearing for the deepeyes Qwen3.5 flow since upstream PRs #8553/#8555 were closed
(unmerged): (1) tool_config_path must be an absolute path; (2) long-prompt prefiltering must be run offline
(prefilter_deepeyes.py) or every startup re-filters for 13-16 min. Neither is in the README; users following
it hit both. Draft a docs PR or comment on #167.
