```{toctree}
:hidden:
:maxdepth: 1

api/overview.md
examples/index.md
claude-skills.md
common-bugs.md
```

# gpCAM

```{div} centered-heading
Autonomous Data Acquisition, HPC Uncertainty Quantification and Constrained Function Optimization
```

```{div} text-center

gpCAM is an API and software designed to make new methodologies for Gaussian process modeling, Bayesian optimization, and decision-making under uncertainty faster, more straightforward, and more widely available. The tool is powered by a flexible and powerful Gaussian process regression at the core. The flexibility stems from the modular design of gpCAM, which allows the user to implement and import their own Python functions to customize and control almost every aspect of the software. Due to a synergy between computational and mathematical function definitions that are fundamental to gpCAM, there are virtually no limits to its customizability. That makes it possible to easily tune the algorithm to account for various types of domain knowledge, consider non-standard inputs, such as distributions and arbitrary structures, and scale it to millions of data points. This makes gpCAM the go-to solution for stochastic function approximation in scientific applications.
```

---

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Simple API
:img-top: _static/laptop-code.jpg

The API calls, function names, and inputs follow the mathematical representation of GPs.
:::

:::{grid-item-card} Powerful Computing
:img-top: _static/contour-plot.png

gpCAM is implemented with options supporting fast training and predictions on HPC architecture.
:::

:::{grid-item-card} Advanced Mathematics for Increased Flexibility
:img-top: _static/surface-plot.png

gpCAM allows virtually unlimited customizability. All functions follow their mathematical definitions; no software limitations are imposed.
:::

:::{grid-item-card} Software for the Novice and the Expert
:img-top: _static/bounded-curve.png

Simple uncertainty quantification and Bayesian optimization problems can be set up in minutes; the options for customization are endless.
:::

:::{grid-item-card} AI Agent Integration
:img-top: _static/ai-agent-integration.png

gpCAM ships as a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) plugin marketplace. AI assistants use the bundled skills to translate plain-English experiment descriptions into runnable gpCAM scripts — no GP expertise required. See [AI Agent Integration](claude-skills.md).
:::

::::

---

```{div} centered-heading
Questions?
```

```{div} text-center

Contact [MarcusNoack@lbl.gov](mailto:MarcusNoack@lbl.gov) to get more information on the project.
```

---

```{div} text-center

gpCAM is a software tool created by CAMERA

The Center for Advanced Mathematics for Energy Research Application
```

```{image} _static/CAMERA_bright.png
:width: 759px
:align: center
```

---

![_static/doe-os.png](_static/doe-os.png)

Supported by the US Department of Energy Office of Science
[Advanced Scientific Computing Research](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) (steven.lee@science.doe.gov)
