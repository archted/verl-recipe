# vllm-ascend: startup crashes on torch>=2.10 because vLLM enables AOT compile by default (aot_compile is not supported ...)

> **渠道**：vllm-project/vllm-ascend issue（蓝海，同事零提交） ｜ **状态**：可提交 ｜ **台账 ID**：I-14 / E-4
> **提交前检查**：确认 nightly/main 是否仍如此；附完整 traceback
> 本文件正文可直接复制粘贴到 GitHub issue/PR。状态变化时改文件名前缀（可提交/待提交）并同步 ARCHED_ISSUES.md。

---

### System Info
- vllm-ascend 0.27.1rc1 (nightly image quay.io/ascend/vllm-ascend:v0.27.1rc1), vLLM 0.27.1, torch 2.10.0+cpu / torch_npu 2.10.0.post4, CANN 9.1.0, Ascend 910B3

### Describe the bug
With torch 2.10, vLLM's `use_aot_compile()` defaults to **enabled** (`VLLM_USE_AOT_COMPILE` defaults to 1 for torch>=2.10 in `vllm/envs.py`). On the vllm-ascend backend the AOT path is not supported, so any rollout/engine start crashes with:

```
RuntimeError: aot_compile is not supported by the current configuration. Please make sure torch.compile is enabled
with the latest version of PyTorch (current using torch: 2.10.0+cpu)
```

This makes the v0.27.1rc1 image unusable out of the box on torch 2.10 stacks — training-side rollout (vLLM engine inside verl) and standalone serving both crash at startup.

### Workaround
`export VLLM_USE_AOT_COMPILE=0` before engine start (plus `--enforce-eager` for the graph-capture path). With it set, PIECEWISE graph capture completes 35/35 and training/serving run normally.

Note for users hitting this via hydra/ray runtime_env: the env var value must be passed as a **string** (`'"0"'`), an int `0` fails later with an unrelated `TypeError: str(dict) expected` deep in ray kwargs parsing.

### Suggested fix
Either default `VLLM_USE_AOT_COMPILE=0` when the ascend platform plugin is active, or fail fast in vllm-ascend with a clear message ("set VLLM_USE_AOT_COMPILE=0") instead of the raw torch aot_compile error.
