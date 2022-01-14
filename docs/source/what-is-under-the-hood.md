---
banner: _static/engine.jpg
banner_brightness: .8
---

# What is Under the Hood

gpCAM is an advanced Gaussian process tool, combined with HPC mathematical optimization. 

Its power comes from the fact that most parts of the Gaussian process
and the steering can be defined by the user as they become more familiar with the underlying mathematics.
One could imagine a car engine; the motor block is the core code of gpCAM,
all other parts that make a well-working car engine are supplied,
but they can be exchanged to create the perfect engine for a particular purpose.
Some of the building blocks that can be defined by the user and imported into gpCAM are:

* A data acquisition function that tells gpCAM where data is sent to and received from

* A plotting function for visualization

* A kernel function to constrain the set of model functions

* A parametric mean function to encapsulate a physics-based model

* An optimizer that replaces the standard optimizers, for instance, for constrained training

* An objective function (or acquisition function) to ingest what patterns the practitioner is looking for

* A cost function to make sure measured points will minimize a cost while maximizing knowledge gain

All this flexibility means that gpCAM naturally includes the ability for physics awareness and also multi-task learning.

Torch and DASK based high-performance computing means that gpCAM can take full advantage of supercomputers.

Read More Here:

* [Advanced Stationary and Non-Stationary Kernel Designs for Domain-Aware Gaussian Processes](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F2102.03432&sa=D&sntz=1&usg=AFQjCNGqDS8i_5Wg1jDCMyGLkZhBMDwuwg)

* [Gaussian processes for autonomous data acquisition at large-scale synchrotron and neutron facilities](https://www.google.com/url?q=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs42254-021-00345-y&sa=D&sntz=1&usg=AFQjCNEfy6-cZzodgWiEIAOHmZ8PxEibfQ)

* [Autonomous materials discovery driven by Gaussian process regression with inhomogeneous measurement noise and anisotropic kernels](https://www.google.com/url?q=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41598-020-74394-1%3Futm_source%3Dother%26utm_medium%3Dother%26utm_content%3Dnull%26utm_campaign%3DJRCN_2_LW01_CN_SCIREP_article_paid_XMOL&sa=D&sntz=1&usg=AFQjCNGGH-Nqdfmm-OaBYrli1BDjj4dF8g)

* [Hybrid genetic deflated Newton method for global optimisation](https://www.google.com/url?q=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS037704271730225X&sa=D&sntz=1&usg=AFQjCNEW5ZKbLA88wAzJDBx6aFDJzX0feQ)

* [Advances in Kriging-Based Autonomous X-Ray Scattering Experiments](https://www.google.com/url?q=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41598-020-57887-x&sa=D&sntz=1&usg=AFQjCNEPz0_JXzvfpKRXsvCu29cNFWfoLw)

* [A Kriging-Based Approach to Autonomous Experimentation with Applications to X-Ray Scattering](https://www.google.com/url?q=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41598-019-48114-3&sa=D&sntz=1&usg=AFQjCNFyVmZu7tigaV2lmbySNHhqpPghhw)


Many of the recent features are not published yet. The papers will be linked here soon.