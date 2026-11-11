# verl: batch-divisibility AssertionError does not explain the constraint or the fix

> **渠道**：verl-project/verl issue ｜ **状态**：可提交 ｜ **台账 ID**：I-23 / E-7（batch 断言半边）
> **提交前检查**：先查重（DX 类可能已有）
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
When launching with a GPU/NPU count that does not divide the batch, verl raises:

```
AssertionError: real_train_batch_size (64) must be divisible by minimal possible batch size (N)
```

The error does not say:
- what "minimal possible batch size" derives from (world size / DP degree),
- which values would be valid (e.g. "use 4, 8 devices for batch 64"),
- that on a multi-node setup `NDEVICES_PER_NODE` vs actual visible devices is the usual culprit.

We hit this when smoke-testing on a subset of cards (6 cards with batch 64): the message gives no path to fix, and the user must bisect device counts by hand. 4 and 8 work, 6 does not.

### Suggested fix
Improve the assertion message to include the effective DP world size and a list of valid batch sizes, e.g.:

```
real_train_batch_size (64) must be divisible by minimal possible batch size (12 = 6 devices × micro 2).
Valid batch sizes for 6 devices: {12, 24, 48, ...}; or change the number of devices to a divisor of 64 (1, 2, 4, 8, 16, 32, 64).
```
