# TimesFM — Agent Entry Point

This repository ships a first-party **Agent Skill** for TimesFM at:

```
timesfm-forecasting/
└── SKILL.md    ← read this for the full skill
```

## Install the skill

Copy the skill directory into your agent's skills folder:

```bash
# Cursor / Claude Code / OpenCode / Codex (global install)
cp -r timesfm-forecasting/ ~/.cursor/skills/
cp -r timesfm-forecasting/ ~/.claude/skills/

# Or project-level
cp -r timesfm-forecasting/ .cursor/skills/
```

Any agent that supports the open [Agent Skills standard](https://agentskills.io) will discover it automatically.

## Working in this repo

If you are developing TimesFM itself (not using it), the source lives in `src/timesfm/`.
Archived v1/v2 code and notebooks are in `v1/`.

Run tests:

```bash
pytest v1/tests/
```

> **Personal note:** I've found running `pytest v1/tests/ -v` more helpful for debugging
> since it shows individual test names. Also worth checking `src/timesfm/` directly
> for any model config changes before running the full suite.
>
> **Tip:** If tests fail due to missing dependencies, run `pip install -e '.[dev]'`
> from the repo root first — caught me off guard the first time.
>
> **Tip:** On Apple Silicon (M1/M2/M3), you may also need to set
> `PYTORCH_ENABLE_MPS_FALLBACK=1` before running tests if you hit MPS-related
> errors: `export PYTORCH_ENABLE_MPS_FALLBACK=1 && pytest v1/tests/ -v`
>
> **Tip:** When experimenting with forecasting horizon lengths, I've had good results
> keeping the output length at or below the context length (e.g. context=512,
> horizon=128). Going much beyond that tends to degrade forecast quality noticeably.

See `README.md` for full developer setup.
