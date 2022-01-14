# gpCAM API Levels

Starting with version 7 of gpCAM, the user has several access points (from high level to low level):

- Using the AutonomousExperimenter functionality
  - AutonomousExperimenterGP: implements an autonomous loop for a single-task GP 
  - AutonomousExperimenterfvGP: implements an autonomous loop for multi-task GP

- The user can use the gpOptimizer (already available in version 6) functionality directly to get full control. The gpOptimizer class is a function optimization wrapper around fvGP, the same is true for the fvgpOptimizer class.
- For GP related work only, the user can use the fvgp package directly (no suggestion capability, no native steering)

For tests and examples, download the repository and go to "./tests".  You will find a "test_notebook" which is probably the best way to get you feed wet and learn about gpCAM. Other tests are available as well though.


Quick Links:

- [Repository](https://github.com/lbl-camera/gpCAM/)
- [AutonomousExperimenter (GP and fvGP)](autonomous-experimenter.md)
- [gpOptimizer](gpOptimizer.md) and [fvgpOptimizer](fvgpOptimizer.md)
- The [fvGP](fvGP.md) Package
- The [HGDL](HGDL.md) Package

```{div} centered-heading
Have suggestions for the API or found a bug?
```

````{div} text-center
Write a brief description of what you saw or what you'd like to see.

```{link-button} https://docs.google.com/forms/d/e/1FAIpQLScaEQSh3585ou8DOJgfPvMaO5I_r_pIVYKRp1ezlJD50-vc4A/viewform
:text: Suggest Now
:classes: btn-primary
```

```{link-button} https://docs.google.com/forms/d/e/1FAIpQLSdTjuO_yTJ_mGFNg7di1GHWzXnEyYHcjVAzrI7M_2Hk3vc0jA/viewform
:text: Report a bug
:classes: btn-primary
```

````
