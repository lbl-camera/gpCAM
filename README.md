# gpCAM

[![PyPI](https://img.shields.io/pypi/v/gpCAM)](https://pypi.org/project/gpcam/)
[![Documentation Status](https://readthedocs.org/projects/gpcam/badge/?version=latest)](https://gpcam.readthedocs.io/en/latest/?badge=latest)
[![gpCAM CI](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml/badge.svg)](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml)
[![Codecov](https://img.shields.io/codecov/c/github/lbl-camera/gpCAM)](https://app.codecov.io/gh/lbl-camera/gpCAM)
[![PyPI - License](https://img.shields.io/badge/license-GPL%20v3-lightgrey)](https://pypi.org/project/gpcam/)
[<img src="https://img.shields.io/badge/slack-@gpCAM-purple.svg?logo=slack">](https://gpCAM.slack.com/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5975552.svg)](https://doi.org/10.5281/zenodo.5975552)
[![Pepy Total Downloads](https://img.shields.io/pepy/dt/gpcam)](https://pepy.tech/project/gpcam)
[![Downloads](https://static.pepy.tech/badge/gpcam/month)](https://pepy.tech/project/gpcam)

> [!WARNING]
> **gpCAM 8.4.0 is a beta release.** It targets `fvgp ~= 4.8` and renames a few
> constructor kwargs (`gp2Scale_dask_client → dask_client`,
> `gp2Scale_linalg_mode → linalg_mode`, `gp_rank_n_update → rank_n_update`) and
> removes `calc_inv` in favor of `linalg_mode="CholInv"`. If you hit issues, pin
> `gpcam==8.3.9` while you report — 8.3.9 remains the stable line. Full migration
> notes are in [HISTORY.rst](HISTORY.rst).



[comment]: <> ([![Maintainability]&#40;https://api.codeclimate.com/v1/badges/29b04c3f69e2b515dac6/maintainability&#41;]&#40;https://codeclimate.com/github/lbl-camera/gpCAM/maintainability&#41;)
[comment]: <> (Hiding maintainibility score while starting to address issues)


gpCAM [(gpcam.lbl.gov)](https://gpcam.lbl.gov/home) is an API and software designed to make advanced Gaussian Process function approximation and autonomous data acquisition/Bayesian Optimization for experiments and simulations more accurate, faster, simpler, and more widely available. The tool is based on a flexible and powerful Gaussian process regression at the core. The flexibility stems from the modular design of gpCAM which allows the user to implement and import their own Python functions to customize and control almost every aspect of the software. That makes it possible to easily tune the algorithm to account for various kinds of physics and other domain knowledge and to identify and find interesting features, in Euclidean and non-Euclidean spaces. A specialized function optimizer in gpCAM can take advantage of HPC architectures for fast analysis time and reactive autonomous data acquisition. gpCAM broke a 2019 record for the largest exact GP ever run! Below you can see a simple example of how to set up an autonomous experimentation loop.


## Usage

The following demonstrates a simple usage of the gpCAM API (see [interactive demo](https://drive.google.com/file/d/1901dKz3ZfcWva1tB9o86IQZT6ukGhV9A/view?usp=sharing)). 

```python
!pip install gpcam

from gpCAM import GPOptimizer

my_gp = GPOptimizer(x_data,y_data,)
my_gp.train()

train_at = [10,20,30] #optional
for i in range(100):
    new = my_gp.ask(np.array([[0.,1.]]))["x"]
    my_gp.tell(new, f1(new).reshape(len(new)))
    if i in train_at: my_gp.train()

```


## Designing experiments with Claude Code

gpCAM ships with a set of [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills that guide an AI assistant through designing autonomous experiments — custom kernels, acquisition functions, noise models, and the full ask/tell/train loop. Experimentalists who want smart, autonomous data acquisition without deep knowledge of GP math or the gpCAM API can use these skills to design autonomous experiments.

### Installing the gpCAM marketplace in Claude Code

The repo ships as a Claude Code [plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces). Inside any Claude Code session, run:

```text
/plugin marketplace add lbl-camera/gpCAM
/plugin install gpcam@gpcam
```

The first command registers this repo as a marketplace; the second installs the `gpcam` plugin from it, which bundles all of the skills below. After install, the skills are available to Claude in any project — no need to clone the repo locally.

To update later, run `/plugin marketplace update gpcam`; to remove, `/plugin uninstall gpcam@gpcam`.

### Available skills

| Skill | Description |
|-------|-------------|
| **experiment-designer** | End-to-end autonomous experiment design. Translates a scientist's description of their measurement into a complete gpCAM script. |
| **kernel-designer** | Design and compose custom kernel functions that encode domain knowledge (smoothness, periodicity, symmetry, anisotropy). |
| **acquisition-functions** | Write custom acquisition functions that encode experimental priorities (exploration vs exploitation, multi-objective, constraints). |
| **prior-mean-functions** | Encode known physics or expected trends as prior mean functions. |
| **noise-functions** | Model position-dependent or heteroscedastic noise from detector characteristics. |
| **cost-functions** | Account for motor travel time, settling, directional costs, and zone-based penalties. |
| **gp2scale-advanced** | Large-scale experiments (>10k points) using sparse kernels and Dask distributed computing. |
| **multi-task-advanced** | Multi-output / function-valued experiments with `fvGPOptimizer`. |
| **transformed-optimizers-advanced** | Constrained observations: `LogGPOptimizer` for `y > 0` and `LogitGPOptimizer` for `y ∈ [0, 1]` — predictions and credible intervals stay inside the constrained range. |

Once installed, the skills activate automatically when you describe an experiment design problem to Claude, or you can invoke one explicitly (e.g. _"use the experiment-designer skill to set up an adaptive XRD scan"_).

### Other agentic platforms

The skills are also compatible with any harness that reads `SKILL.md` files (e.g. [OpenClaw](https://openclaw.ai)) — clone the repo and point your assistant at the `skills/` directory. When this repo is present in your working directory, Claude Code also picks up the root `CLAUDE.md` and `skills/` directory automatically, so the marketplace install is only needed for use outside the repo.


## Credits

Main Developer: Marcus Noack ([MarcusNoack@lbl.gov](mailto:MarcusNoack@lbl.gov))

This code was developed with help from Ron Pandolfi (LBNL), Mark Risser (LBNL), Hengrui Luo (Rice U.), and Vardaan Tekriwal (UCB).

Additional contributions and insights came from across the community, in particular, Kevin Yager, Masafumi Fukuto, and their teams (Brookhaven National Lab).

We acknowledge support from several DOE ASCR, BER, and BES projects, including CAMERA (James Sethian, Jeff Donatelli), SPECTRA (Sherry Li), and CASCADE (Bill Collins), as well as support directly from Lawrence Berkeley National Laboratory.

This package uses the HGDL package of David Perryman and Marcus Noack, which is based on the HGDN algorithm by Noack and Funke.



