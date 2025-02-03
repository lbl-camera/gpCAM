---
banner: _static/landing.png
banner_height: "60vh"
banner_contents: >
    <div style="position: absolute; top: 50%; left: 60%; transform: translate(-50%, -50%); color: white;"><p style="text-align: center;"><em id="landing-title" style="font-size: 48pt; color: #96e6b3; font-family: \'Merriweather\', serif; font-weight: 900; font-style: italic;">gpCAM</em></p>
    <p style="text-align: center;"><em>Autonomous Data Acquisition, HPC Uncertainty Quantification and Constrained Function Optimization</em></p>
    </div>
---

```{toctree}
---
hidden: true
maxdepth: 2
caption: API
---
api/index.md
api/autonomous-experimenter.md
api/autonomous-experimenterfvgp.md
api/gpOptimizer.md
api/fvgpOptimizer.md
api/gpMCMC.md
api/logging.md
```


```{toctree}
---
hidden: true
maxdepth: 1
caption: Examples
---
examples/autonomous_experimenter_basic.ipynb
examples/autonomous_experimenter_advanced.ipynb
examples/1dSingleTaskAcqFuncTest.ipynb
examples/GPOptimizer_SingleTaskTest.ipynb
examples/GPOptimizer_NonEuclideanInputSpaces.ipynb
examples/GPOptimizer_gp2ScaleTest.ipynb
examples/GPOptimizer_MultiTaskTest.ipynb
examples/GPOptimizer_Optimization.ipynb
```


# gpCAM

```{div} centered-heading
Mission of the project
```

```{div} text-center

gpCAM is an API and software designed to make new methodologies for Gaussian process modeling, Bayesian optimization, and decision-making under uncertainty faster, more straightforward, and more widely available. The tool is powered by a flexible and powerful Gaussian process regression at the core. The flexibility stems from the modular design of gpCAM, which allows the user to implement and import their own Python functions to customize and control almost every aspect of the software. Due to a synergy between computational and mathematical function definitions that is fundamental to gpCAM, there are virtually no limits to its customizability. That makes it possible to easily tune the algorithm to account for various types of domain knowledge, consider non-standard inputs, such as distributions and arbitrary structures, and scale it to millions of data points. This makes gpCAM the go-to solution for stochastic function approximation in scientific applications. 
```

---

``````{div} container card-box
`````{div} row

```` {div} col w-200
![_static/laptop-code.jpg](_static/laptop-code.jpg)
````

````{div} col
```{div} h3
Simple API
```

The API calls, function names, and inputs follow the mathematical representation of GPs.
````
`````
````` {div} row
```` {div} col
![_static/contour-plot.png](_static/contour-plot.png)
````

```` {div} col
```{div} h3
Powerful Computing
```

gpCAM is implemented with options supporting fast training and predictions on HPC architecture.

````
`````
````` {div} row
```` {div} col
![_static/surface-plot.png](_static/surface-plot.png)  
````

```` {div} col
```{div} h3
Advanced Mathematics for Increased Flexibility
```

gpCAM allows virtually unlimited customizability; all functions follow their mathematical definitions. There are absolutely no limitations imposed by the software design.

````
`````
````` {div} row
```` {div} col
![_static/bounded-curve.png](_static/bounded-curve.png)
````

```` {div} col
```{div} h3
Software for the Novice and the Expert
```

Simple uncertainty quantification and Bayesian optimization problems can be set up in minutes; the options for customization are endless.

````
`````
``````

---

```{div} centered-heading
Questions?
```

````{div} text-center

Contact [MarcusNoack@lbl.gov](mailto:MarcusNoack@lbl.gov) to get more information on the project. 

---

gpCAM is a software tool created by CAMERA

The Center for Advanced Mathematics for Energy Research Application

```{image} _static/CAMERA_bright.png
:width: 759px
```
````

---


![_static/doe-os.png](_static/doe-os.png)

Supported by the US Department of Energy Office of Science  
[Advanced Scientific Computing Research](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) (steven.lee@science.doe.gov)
````



