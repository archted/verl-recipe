# opencode: legacy per-model `maxOutputTokens`/`maxInputTokens` config keys are silently ignored (v2 uses `limit`)

> **渠道**：opencode (sst/opencode) issue ｜ **状态**：可提交 ｜ **台账 ID**：I-18 / E-9
> **提交前检查**：确认最新版 schema；附最小复现配置
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug
opencode v2.2.5 config schema expects per-model limits under:

```json
{ "options": { "baseURL": "...", "apiKey": "...", "limit": { "context": 8192, "output": 4096 } } }
```

The v1-style keys `maxInputTokens` / `maxOutputTokens` are **silently ignored** — no validation error, no warning. The model then defaults to a huge completion budget (e.g. 32000), and against a strict backend (local vLLM with `max_model_len=8192`) every request fails with:

```
max_completion_tokens=32000 cannot be greater than max_model_len=8192
```

while the config plainly set `maxOutputTokens: 8192`. Against permissive gateway models the mistake is invisible, which makes it worse: the same config works in one setup and breaks in another with no hint.

### Suggested fix
1. Emit a config-validation warning when legacy keys are present ("maxOutputTokens is ignored; use limit.output").
2. Better: accept the legacy keys as aliases.

### Repro
opencode.config model entry with `maxOutputTokens: 8192` → point provider at a vLLM server with `--max-model-len 8192` → request fails with max_completion_tokens=32000.
