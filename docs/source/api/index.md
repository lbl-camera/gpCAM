# gpCAM API Levels

Starting with version 7 of gpCAM, the user has several access points (from high level to low level):

- Using the AutonomousExperimenter functionality
  - AutonomousExperimenterGP: implements an autonomous loop for a single-task GP 
  - AutonomousExperimenterfvGP: implements an autonomous loop for multi-task GP

- The user can use the gpOptimizer (already available in version 6) functionality directly to get full control. The gpOptimizer class is a function optimization wrapper around fvGP, the same is true for the fvgpOptimizer class.
- For GP related work only, the user can use the fvgp package directly (no suggestion capability, no native steering)

For tests and examples, check out the examples on this very website or download the repository and go to "./tests".


Quick Links:

- [Repository](https://github.com/lbl-camera/gpCAM/)
- [AutonomousExperimenter (GP and fvGP)](autonomous-experimenter.md)
- [gpOptimizer](gpOptimizer.md) and [fvgpOptimizer](fvgpOptimizer.md)
- The [fvGP](https://fvgp.readthedocs.io/en/latest/index.html) Package
- The [HGDL](https://hgdl.readthedocs.io/en/latest/index.html) Package

```{div} centered-heading
Have suggestions for the API or found a bug? 
```

````{div} text-center
Please submit an issue on [GitHub](https://github.com/lbl-camera/gpCAM/).
````
