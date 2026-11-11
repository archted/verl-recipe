# PR：[recipe/deepeyes] prepare_data.py: add --vstar-only to exclude no-ground-truth sources

> **渠道**：verl-project/verl-recipe PR ｜ **状态**：可提交 ｜ **台账 ID**：I-16
> **提交前检查**：代码已就绪：本仓 commit a7c10a2（scripts/prepare_data.py）；datasets 5.1.0 filter 实测可用；建议与“无GT样本不告警”issue 一同提
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

**PR title:** `[recipe/deepeyes] prepare_data: add --vstar-only flag; report no-GT fraction`

```markdown
## Problem
The DeepEyes training mix includes sources without gradeable ground truth (thinklite: 10,953 of 33,137
samples in our set). These samples can never earn acc reward, consume full rollout/judge/training compute,
and dilute GRPO group baselines — with no warning anywhere.

## Change
1. `prepare_data.py` prints the no-ground-truth fraction per source after preprocessing.
2. New `--vstar-only` flag filters `data_source == "vstar"` for users who want a single-source run.

## Validation
`--help` smoke-tested; `Dataset.filter(input_columns=...)` verified on datasets 5.1.0; used to prepare our
next training round.
```
