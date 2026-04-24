# gpCAM

[![PyPI](https://img.shields.io/pypi/v/gpCAM)](https://pypi.org/project/gpcam/)
[![Documentation Status](https://readthedocs.org/projects/gpcam/badge/?version=latest)](https://gpcam.readthedocs.io/en/latest/?badge=latest)
[![gpCAM CI](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml/badge.svg)](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml)
[![Codecov](https://img.shields.io/codecov/c/github/lbl-camera/gpCAM)](https://app.codecov.io/gh/lbl-camera/gpCAM)
[![PyPI - License](https://img.shields.io/badge/license-GPL%20v3-lightgrey)](https://pypi.org/project/gpcam/)
[<img src="https://img.shields.io/badge/slack-@gpCAM-purple.svg?logo=slack">](https://gpCAM.slack.com/)
[![DOI](https://zenodo.org/badge/434768487.svg)](https://zenodo.org/badge/latestdoi/434768487)
![Downloads](https://img.shields.io/pypi/dm/gpcam)


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

When you clone this repo, the root `CLAUDE.md` and `skills/` directory are picked up automatically by Claude Code. Available skills:

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

These skills are also compatible with other agentic platforms (e.g. [OpenClaw](https://openclaw.ai), or any harness that can read `SKILL.md` files) — point your assistant at the `skills/` directory.


## Credits

Main Developer: Marcus Noack ([MarcusNoack@lbl.gov](mailto:MarcusNoack@lbl.gov))

This code was developed with help from Ron Pandolfi (LBNL), Mark Risser (LBNL), Hengrui Luo (Rice U.), and Vardaan Tekriwal (UCB).

Additional contributions and insights came from across the community, in particular, Kevin Yager, Masafumi Fukuto, and their teams (Brookhaven National Lab).

We acknowledge support from several DOE ASCR, BER, and BES projects, including CAMERA (James Sethian, Jeff Donatelli), SPECTRA (Sherry Li), and CASCADE (Bill Collins), as well as support directly from Lawrence Berkeley National Laboratory.

This package uses the HGDL package of David Perryman and Marcus Noack, which is based on the HGDN algorithm by Noack and Funke.



