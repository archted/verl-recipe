# opencode: fixed managed-service port 49374 conflicts under host-network containers; no auto-fallback

> **渠道**：opencode (sst/opencode) issue ｜ **状态**：待提交 ｜ **台账 ID**：I-21 / E-8
> **提交前检查**：查重 opencode 仓是否已有端口回退机制；准备复现步骤
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### Describe the bug (draft)
opencode's managed service binds a fixed default port (49374). Running opencode inside a `--net=host` container
on a host that also runs opencode fails with "Managed service port 49374 on 127.0.0.1 is already in use" —
the user must know to run `opencode service set port 49555`. Suggest: honor an env var (e.g. OPENCODE_SERVICE_PORT)
and/or auto-fallback to the next free port with a clear log line.
