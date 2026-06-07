# CLAUDE.md

## Purpose
This folder is a sandbox for experimenting with **Claude Code skills and agent configurations**. Treat it as a scratch area for prototyping — not a production project.

## Project Lessons (consult first)
- **Before non-trivial pipeline work (esp. Stage 3), READ `tasks/lessons.md`.** It holds project-local, reusable lessons (e.g. the Stage 3 orchestrator `--resume` loop gotcha + absolute `BLENDER` requirement). Update it after each task per the Task Management workflow below.

## Directory Conventions

### `./tmp/` — One-off Python code
- All one-off / throwaway Python scripts MUST be saved under `./tmp/`.
- Do NOT scatter ad-hoc scripts in the repo root or elsewhere.
- Contents of `./tmp/` are expected to be deleted after the experiment is done. Do not rely on anything inside it persisting.

### `.claude/skills/` — Skill definitions
- When creating a new skill, the skill's Python file(s) MUST live under `.claude/skills/<skill-name>/`.
- Every skill needs a `SKILL.md` in its folder describing trigger conditions and usage.
- Do NOT place skill Python files anywhere else (e.g., not in repo root, not in `./tmp/`).

### `.claude/agents/` — Agent definitions
- Any custom subagent markdown files go under `.claude/agents/`.

## Cleanup Rule
- After an experiment is finished, `./tmp/` should be removed (or emptied).
- Skills and agents that proved useful stay under `.claude/`; everything else in `./tmp/` is disposable.

---

# Rules for Claude

## Language
- Always respond in **ENGLISH**.

# Workflow Orchestration

## 1. Plan Node Default
- Enter plan mode for all non-trivial tasks (3+ steps or architectural decisions).
- If blocked, stop immediately and replan.
- Use plan mode for both implementation and validation.
- Write detailed specs beforehand to reduce ambiguity.

## 2. Subagent Strategy (MANDATORY — TOP PRIORITY)

> Principle: "Do not do it yourself. Delegate."

The main agent acts only as an **orchestrator**. All execution must be done by **subagents**.

### 2.1 Mandatory Subagent Use (No Exceptions)
- Code exploration/search → `Explore`
- Code analysis/understanding → `code-analyzer`
- Web research → `web-searcher-learner`
- Implementation → `meshlayout-coder` or `general-purpose`
- Planning → `careful-planner` or `Plan`
- Review/validation → appropriate agent

### 2.2 Allowed Direct Handling (Only Exceptions)
- 1–2 line change in a single file
- Simple Q&A
- Single `Read`, `Glob`, `Grep` usage

### 2.3 Execution Rules
- Run independent subagents in parallel (no sequential execution).
- One focused task per **subagent**.
- Split implementation by file/module for parallel execution.
- Use `run_in_background` for long-running tasks.
- Provide full context (paths, requirements, constraints).

### 2.4 Self-Check
- Reading >3 lines of code → delegate
- Exploring multiple files → use Explore agent
- Guessing instead of searching → use web-searcher
- About to implement directly → stop and delegate
- Main agent = planning + coordination only

## 4. Verification Before Done
- Do not mark complete without proof of functionality
- Compare behavior before/after changes
- Ask: "Would a senior engineer approve this?"
- Run tests, check logs, verify correctness

---

# Task Management

1. Plan in `tasks/todo.md`
2. Get user confirmation
3. Track progress with checkboxes
4. Provide high-level summaries
5. Add review section in `tasks/todo.md`
6. Update `tasks/lessons.md`

---

# Core Principles

- **Simplicity First**: Minimize complexity and impact.
- **No Laziness**: Fix root causes, no temporary hacks.
- **Minimal Impact**: Only change what is necessary.
- **Minimal Fix Over New Components**: When a single component is the root cause of a bug, fix that component — do NOT propose adding a new agent/hint/component on top of logic that already works. Before proposing a new component (agent hint, validator, extra stage), confirm the existing logic is actually broken. If the existing computation is correct and a downstream guard/clamp/filter is what's corrupting it, fix that guard — don't bolt on a new "smart" layer. Also verify claims about what already exists before recommending additions.
  - *Why:* In the kinder_new world-scale bug, the class-prior algorithm already computed the correct raw k=9.70; only the `[0.5, 2.0]` clamp broke it (clamped to 2.0). The minimal correct fix was the clamp alone — not a new vision-agent `scale_hint`.
