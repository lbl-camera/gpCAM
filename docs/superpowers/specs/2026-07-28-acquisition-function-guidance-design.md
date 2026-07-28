# Acquisition-function guidance in the gpCAM skills

**Date:** 2026-07-28
**Status:** approved

## Problem

The gpCAM Claude skills lead users to `"expected improvement"` when another acquisition
function would serve their experiment better. This is not a single bad sentence; it is
four reinforcing pressures plus a silent gap.

### 1. The experiment-designer funnels into it

`skills/experiment-designer/SKILL.md` Step 1 asks the scientist:

> **Goal**: Exploration (map everything)? Optimization (find the peak)? Both?

Nearly every beamline scientist answers "find the peak." That answer lands on the
optimization branch of the design table (line 45):

> `'variance'` for exploration/mapping. `'expected improvement'` or `'ucb'` for
> optimization

EI is first. The script template's option comment (line 166) repeats the ordering.

### 2. EI has the broadest "Best for" cell

In `skills/acquisition-functions/SKILL.md` the built-in table gives EI
*"Optimization (find max)"* — the most generic description in the table. UCB reads
*"Maximization with tunable exploration/exploitation"*, which sounds like a knob the
reader must first understand; `"gradient"`, `"target probability"` and
`"relative information entropy"` all read as narrow specialties. EI is the
lowest-effort pick.

### 3. Nothing says when *not* to use it

Neither skill lists a single EI failure mode, and neither instructs the agent to work
the choice out with the scientist. The decision is presented as a lookup, not a
conversation.

### 4. Model prior

EI is the most-cited acquisition function in the Bayesian-optimization literature.
Absent explicit counter-guidance, an LLM defaults to it. Silence in the skills is
sufficient cause.

## gpCAM's EI does not behave the way the skills imply

Five behaviors are undocumented in the skills. They are the substantive reason this
matters — a user told "EI exploits toward the maximum" is being misinformed.

**a. Improvement is clipped before the ratio** (`gpcam/surrogate_model.py:254`):

```python
a = (m - last_best).reshape(len(x))
a[a < 0.] = 0.              # clipped BEFORE forming gamma
gamma = a / (std + 1e-9)
return std * (gamma * cdf + pdf)
```

Textbook EI clips after forming `z = (m - y_best)/sigma`. Here, at every candidate whose
posterior mean sits below the incumbent, `gamma == 0`, so the score collapses to
`std * norm.pdf(0) = 0.3989 * std` — which is the `"variance"` acquisition up to a
positive constant. Early in a run, when nothing beats the incumbent, **gpCAM's EI is
pure exploration** and only begins to differentiate once the posterior mean exceeds the
best observed value.

**b. Maximization only.** `last_best = np.max(gpo.y_data)` is hardcoded
(`surrogate_model.py:257`). A scientist minimizing gets silently meaningless behavior.

**c. The incumbent is a noisy observation,** not the posterior mean at the incumbent.
With noisy data EI anchors to a noise spike and over-exploits around it.

**d. The multi-task branch is dimensionally incoherent** (`surrogate_model.py:306`): it
sums the posterior mean across tasks and *sums* the standard deviations across tasks,
then compares that sum against `np.max(gpo.y_data)` of a scalar observation.

**e. Batch silently rewrites it.** `ask(n>1)` without `method="hgdl"` and with a string
acquisition replaces the request with `"total correlation"`
(`gpcam/gp_optimizer_base.py:517-525`).

## Separate factual error found

`skills/experiment-designer/SKILL.md:45` claims UCB "exposes a tunable
exploration/exploitation tradeoff via `beta`". The built-in `"ucb"` string hardcodes
`beta = 3.0` (`surrogate_model.py:231`). Only a custom callable is tunable.

## Design

### Files changed

**1. `skills/acquisition-functions/SKILL.md`** — restructured.

A new section ahead of the built-in table reframes the choice as a collaboration:
the agent must name its recommendation, state the one tradeoff that recommendation
makes, and confirm with the scientist before generating a script. "Find the best
conditions" is explicitly called out as ambiguous — the agent must ask whether they
want the single best point or a trustworthy map that contains it.

A decision table keyed on *what the scientist wants to learn* replaces the
exploration-vs-optimization binary. Rows cover: map the space, find where the signal
changes fastest, find the single best point on a tight budget, find a threshold
crossing, hit a target value, minimize, batch, multi-task. EI keeps a row, honestly
scoped to late-stage refinement of a maximum on low-noise sequential single-task data.

The built-in table gains an **"Avoid when"** column.

A new **"gpCAM-specific behavior you must know"** section documents (a)-(e) above with
source-line citations.

A new recipe is added beside the existing ones:

```python
def radical_gradient(x, gpo):
    g = gpo.posterior_mean_grad(x)["dm/dx"]
    std = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"])
    return np.sqrt(np.linalg.norm(g, axis=1)) * std
```

The built-in `"gradient"` is `||grad m|| * sigma`; the radical form takes the square root of
the gradient term only, softening the gradient weighting so uncertainty carries
relatively more weight. Sampling then spreads more broadly across the space instead of
piling onto the single steepest ridge. Preferred for mapping *changes* through
parameter space; the built-in `"gradient"` remains a good alternative.

**2. New `skills/acquisition-functions/references/choosing-an-acquisition-function.md`**

One worked example per experiment archetype: peak finding, mapping / surrogate
building, boundary or threshold finding, target value, mapping change (gradient vs
radical gradient, and when each wins), minimization, batch, multi-task, and
cost-constrained. Each entry gives how the scientist typically phrases the goal, what
to ask back, the resulting choice, and the loop snippet.

**3. `skills/experiment-designer/SKILL.md`**

Step 1's goal question becomes the archetype list rather than the binary. The design
table row defers to the acquisition skill instead of naming a winner. The template
comment keeps `"variance"` as the safe default but marks it as requiring confirmation.
The UCB `beta` claim is corrected.

**4. `skills/multi-task-advanced/SKILL.md`** — a warning against EI for multi-task.

**5. `CLAUDE.md`** — a new entry under "Key principles for generated experiment
scripts": confirm the acquisition function with the user; do not default to EI.

### Out of scope

`gpcam/surrogate_model.py` is not modified. The pre-ratio clipping in EI is arguably an
upstream bug, but changing acquisition math is a separate decision from correcting the
skills. This spec flags it; it does not fix it.
