# PR：[recipe/deepeyes] fix reward fn: skip judge when reward is already decided; richer failure logs

> **渠道**：verl-project/verl-recipe PR（关联 #166/#167） ｜ **状态**：可提交 ｜ **台账 ID**：I-1 / E-10 / DTS-002
> **提交前检查**：代码已就绪：本仓 commit 9c56e7f（deepeyes.py）；提 PR 前签 CLA + 先在 #167 评论
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

**PR title:** `[recipe/deepeyes] reward fn: don't call the judge for samples already scored 0; log judge failure bodies`

**关联 issue：** judge 400 issue（先提 issue 再链 PR）

**What & why（正文英文可直接粘贴）：**

```markdown
## Problem
`compute_score` calls the LLM judge **before** the `len(answer_text) >= 1000 → acc=0` anti-hacking guard.
Any malformed rollout (missing `</answer>`) falls back to using the whole response tail (up to the full
16k-token rollout) as the "answer", overruns the judge context (max_model_len=8192) and logs one HTTP 400
per malformed rollout — 87 failures per 2048 rollouts in our run (~4%, matching the format-error rate).
Those samples are scored 0 by the length rule regardless of the verdict, so every 400 is a wasted judge
call (3 retries + ~16k-token prefill) and pure log noise that masked the real signal (format-error rate).

## Measurements (judge's own tokenizer, full training set)
- worst legitimate prompt = few-shot 751 + longest question+GT 394 + 999-char answer 615 = **1760 tokens**
  vs the 8176-token budget → legitimate traffic never approaches the limit
- judge KV cache at TP=8: 2,049,536 tokens (8192 uses 0.4%)

## Change
1. Move the `len(answer_text) >= 1000 → acc=0, format=-1` check **before** `_judge_accuracy`:
   400s drop to zero, judge load drops ~4%, reward values are bit-identical.
2. Log the judge response body + prompt_chars on failure (previously only "400 Client Error", which
   led to a misdiagnosis and a harmful truncation patch that we reverted — truncating ground_truth,
   up to 1884 chars in our data, can flip CORRECT to INCORRECT).

## Tests
4 unit cases run in-container: no-<answer> 5000-char tail → 0 judge calls; short answer → 1 call, acc 1;
1200-char answer → 0 calls, acc 0, format -1; judge 400 → no crash, log contains body + prompt_chars.
```

**代码位置：** 本仓 `deepeyes-qwen35-npu` 分支 commit `9c56e7f`（deepeyes.py，两份副本逐字节一致）。
