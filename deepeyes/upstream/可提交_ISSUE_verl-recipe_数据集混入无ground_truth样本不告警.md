# verl-recipe (deepeyes): dataset silently mixes samples without ground_truth; they can never earn reward but consume full training compute

> **渠道**：verl-project/verl-recipe issue ｜ **状态**：可提交 ｜ **台账 ID**：I-16
> **提交前检查**：查重；附 prepare_data 统计输出
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
The DeepEyes recipe's data preparation (and the verl dataloader) accept a training set where a large fraction of samples have **no usable ground truth**. In our run, 10,953 of ~33k training samples (a `thinklite` subset) carry no gradeable answer: the reward function can only ever return `format`-only scores, `acc_reward` is structurally 0 for them, and every one of their 8 GRPO rollouts consumes full rollout + judge + training compute while contributing nothing to the accuracy signal. The dataloader/trainer raises no warning about the no-GT fraction.

### Evidence
- `prepare_data.py` statistics on our preprocessed set: 33,137 train samples; thinklite entries with empty/unusable `reward_model.ground_truth` = 10,953.
- Training log: `critic/score/mean` plateaus while `response_length` collapses — mixed no-GT data dilutes the advantage signal for the gradeable subset.
- A reward-hacking-adjacent risk: samples with no GT get score only from `format + tool` channels, so the cheapest way to score on them is short format-correct outputs — this plausibly contributed to our policy collapse (short-answer mode, see related zero-variance issue).

### Suggested fix
1. In `prepare_data.py` (or the dataset class), compute and **print the no-GT fraction**; add a flag (we implemented `--vstar-only` locally to filter `data_source == "vstar"`) to exclude them.
2. In the trainer, WARN once per run when a large fraction of samples cannot earn `acc` reward.

### Impact
~1/3 of our training compute was spent on samples that cannot learn accuracy; this also biases GRPO group baselines when a group mixes GT and no-GT samples.
