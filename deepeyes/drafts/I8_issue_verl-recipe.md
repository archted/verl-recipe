# [DRAFT issue] verl-project/verl-recipe

**Title:** [deepeyes] `run_deepeyes_grpo.sh` fails out of the box: dead `REF_MODEL_PATH` placeholder is passed as the actor model path

**Labels 建议加:** deepeyes / recipe

---

### System Info
- Repo: verl-project/verl-recipe (main, file `deepeyes/run_deepeyes_grpo.sh`, introduced by the recipe migration PR #7)
- OS: Linux (Ascend NPU 8x910B3 environment, but this is host-side script logic, platform independent)

### Describe the bug

Running `deepeyes/run_deepeyes_grpo.sh` as-is fails at startup because of a never-assigned placeholder:

```bash
# line 16
REF_MODEL_PATH=ref_model_path
# line 30
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
```

`ref_model_path` is a literal string that nothing in the script (or docs) tells the user to replace, and it is passed to `actor_rollout_ref.model.path`. Hydra/verl then tries to load the model from the relative directory `ref_model_path`, which does not exist, so the run crashes before training starts.

Two additional problems beyond the crash:

1. **Misleading name.** In verl, `actor_rollout_ref.model.path` configures the **actor/rollout base model**, not a reference policy. This recipe disables the ref model anyway (`algorithm.kl_ctrl.kl_coef=0.0`, `actor_rollout_ref.actor.use_kl_loss=False`), so no ref weights are ever loaded. A user who dutifully fills in their *reference* model path here would silently train the wrong model.
2. **Stale paths from the migration.** The script was migrated from `verl` (PR #7) but still uses the old layout: `--config-path=${BASEDIR}/recipe/deepeyes/configs`, while in verl-recipe the recipe lives under `deepeyes/` (e.g. `deepeyes/configs/image_zoom_in_tool_config.yaml` at line with `tool_config_path` uses the new layout, the `--config-path` does not). Users must figure this out by trial and error.

### Steps to reproduce

1. Clone verl-recipe, cd into it.
2. Set `BASEDIR` to any directory, run `bash deepeyes/run_deepeyes_grpo.sh` (with the other obvious placeholders filled or not — the crash below happens regardless).
3. Startup fails resolving the model path `ref_model_path`.

### Expected behavior

The script should either run out of the box with a sensible default model, or fail fast with a clear message telling the user which variable to set.

### Proposed fix (PR incoming)

Rename to `MODEL_PATH` and make it a guarded default so it is both runnable and overridable:

```bash
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
...
    actor_rollout_ref.model.path=${MODEL_PATH} \
```

plus a short comment clarifying that `actor_rollout_ref.model.path` is the actor/rollout model and that this recipe never loads a separate ref policy (KL disabled).

I have a one-file PR ready (will link it here once submitted) — happy to also fix the stale `--config-path` in the same PR if maintainers prefer, otherwise I will keep it minimal.
