# AI Agent Integration

gpCAM ships with a set of [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
skills that guide an AI assistant through designing autonomous experiments —
custom kernels, acquisition functions, noise and cost models, the full
`ask` / `tell` / `train` loop, and large-scale or multi-task setups.

Experimentalists who want smart, autonomous data acquisition without deep
knowledge of GP math or the gpCAM API can let an AI assistant translate a
plain-English description of their measurement into a working gpCAM script.

## Installing the marketplace in Claude Code

The gpCAM repository is published as a Claude Code
[plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces).
Inside any Claude Code session, run:

```text
/plugin marketplace add lbl-camera/gpCAM
/plugin install gpcam@gpcam
```

The first command registers this repo as a marketplace and pulls its
`marketplace.json` manifest. The second installs the `gpcam` plugin from
that marketplace — a single plugin that bundles every skill listed below.
Once installed, the skills are available to Claude in any project on your
machine; you do **not** need to clone gpCAM locally.

Useful follow-up commands:

| Command | What it does |
|---|---|
| `/plugin marketplace list` | Show the marketplaces you've added. |
| `/plugin marketplace update gpcam` | Pull the latest skill versions. |
| `/plugin list` | Show installed plugins. |
| `/plugin uninstall gpcam@gpcam` | Remove the plugin. |

## Using the skills

Once the plugin is installed, Claude will activate the appropriate skill
automatically when you describe an experiment-design problem. For example:

> *"I want to map the photoluminescence of a thin film over a 2-inch wafer.
> Each measurement takes 30 seconds, motor moves are slow in the X direction,
> and I have 4 hours of beam time."*

…will trigger the **experiment-designer** skill (and pull in
**cost-functions** for the asymmetric motor cost). You can also invoke a
skill explicitly:

> *"Use the kernel-designer skill to build a periodic + Matérn kernel for
> a temperature-dependent diffraction scan."*

## Available skills

```{list-table}
:header-rows: 1
:widths: 25 75

* - Skill
  - Description
* - **experiment-designer**
  - End-to-end autonomous experiment design. Translates a scientist's
    description of their measurement into a complete, runnable gpCAM script.
* - **kernel-designer**
  - Design and compose custom kernel functions that encode domain knowledge
    — smoothness, periodicity, symmetry, anisotropy, non-Euclidean inputs.
* - **acquisition-functions**
  - Write custom acquisition functions that encode experimental priorities:
    exploration vs exploitation, multi-objective targets, constraints,
    cost-aware acquisition, UCB / LCB, probability of improvement.
* - **prior-mean-functions**
  - Encode known physics or expected trends as prior mean functions, so the
    GP regresses against a baseline rather than a flat zero prior.
* - **noise-functions**
  - Model position-dependent, heteroscedastic, or count-rate-dependent noise
    from detector characteristics.
* - **cost-functions**
  - Account for motor travel time, settling, directional costs, sample
    damage, beam time, and zone-based penalties.
* - **gp2scale-advanced**
  - Large-scale experiments (>10k points up to millions) using sparse,
    compactly-supported kernels and Dask distributed computing.
* - **multi-task-advanced**
  - Multi-output, vector-valued, or function-valued experiments using
    `fvGPOptimizer`.
```

## Using the skills outside Claude Code

The skills are plain Markdown `SKILL.md` files and are compatible with any
agentic harness that can read them — for example
[OpenClaw](https://openclaw.ai). To use them outside Claude Code:

1. Clone the gpCAM repository:

   ```bash
   git clone https://github.com/lbl-camera/gpCAM.git
   ```

2. Point your assistant at the `skills/` directory at the repo root.

When the cloned repo is present in your working directory, Claude Code
also picks up the root `CLAUDE.md` and `skills/` directory automatically,
so the marketplace install is only needed for use *outside* the repo.

## Source

The marketplace manifest lives at
[`.claude-plugin/marketplace.json`](https://github.com/lbl-camera/gpCAM/blob/main/.claude-plugin/marketplace.json)
and the per-skill content under
[`skills/`](https://github.com/lbl-camera/gpCAM/tree/main/skills).
Contributions and new skills are welcome — open a pull request.
