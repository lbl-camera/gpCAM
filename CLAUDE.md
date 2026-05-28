# gpCAM Claude Skills

You are helping scientists design autonomous experiments using gpCAM, a Gaussian Process-based Bayesian optimization toolkit.

## Skills

Read the appropriate skill file before generating code:

- **Designing an experiment**: `skills/experiment-designer/SKILL.md`
- **Custom kernels**: `skills/kernel-designer/SKILL.md`
- **Acquisition functions**: `skills/acquisition-functions/SKILL.md`
- **Prior mean functions**: `skills/prior-mean-functions/SKILL.md`
- **Noise models**: `skills/noise-functions/SKILL.md`
- **Cost functions (travel time)**: `skills/cost-functions/SKILL.md`
- **Large-scale (>10k points)**: `skills/gp2scale-advanced/SKILL.md`
- **Multi-task/multi-output**: `skills/multi-task-advanced/SKILL.md`

## Reference Materials

- [gpCAM documentation](https://gpcam.readthedocs.io) — full API reference and mathematical background
- `gpcam/kernels.py` — re-exports the fvGP kernel library (`from fvgp.kernels import *`)

## Key Principles

1. **Generate complete, runnable scripts** — not fragments
2. **Target audience is scientists**, not GP experts — explain choices in plain language
3. **Always document the hyperparameter layout** — which index maps to what
4. **Hyperparameter bounds must match** the total hyperparameter count across kernel + mean + noise
5. **Default kernel is usually fine** — only suggest custom when there's a clear reason
6. **Use vectorized numpy** — no Python loops over data points
7. **Return dict keys**: `posterior_mean()` returns `"m(x)"`, `posterior_covariance()` returns `"v(x)"` and `"S"`. NOT `"f(x)"`.
8. **`get_data()` keys use spaces**: `"x data"`, `"y data"`, `"hyperparameters"`, `"measurement variances"`. NOT underscores.
