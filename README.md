# MultiFidelity_Bayes
Manuscript for first EAGER grant paper on Bayesian inference driven force field parameterization using a multi-fidelity likelihood calculation algorithm

Test container to display in-progress machinery for using amortized inference to refit the 4-D C-C & C-H LJ parameters (to scale up) from the `smirnoff99Frosst` forcefield. We are training solely on molar volume and heat of vaporization from cyclohexane.

# Repositories
- `/scripts`: contains all python scripts and ipython notebooks demonstrating making estimates of properties using `pymbar`, constructing Gaussian process regressions with `gpy` and implementing MCMC using `emcee` in order to sample from a posterior distribution of forcefield parameters.
- `/simulated_estimates`: contains observables calculated by simulation at ~120 different parameter states in order to compare to MBAR.
- `/MBAR_estimates`: contains statistically robust observables calculated by MBAR used to construct Gaussian Process regressions.
- `/figures`: contains figures constructed to visually debug simulated energy distributions.
