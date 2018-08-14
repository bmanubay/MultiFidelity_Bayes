This folder contains the code I use for making and distributing MBAR calculations on Lilac as well as an notebook highlighting the process for making GPs and using them for posterior sampling (messy).


# MBAR Stuff
`run_MBAR_arbitrary_dims.py` - file used to make MBAR calculations given input reference(s) and an arbitrary dimension of parameter changes

`merge_estimates.py` - used to combine all of the estimates from files saved from the distributed calculations
`4D_MBAR_estimates.lsf` - example lsf script for distributing MBAR caclulations efficiently on Lilac

# Simulation Stuff
`run_molecule_4D.py` - run bulk simulations with 4D parameter changes in LJ SMIRKS that can be specified (a lot of hardcoding. can be made more modular)
`run_molecule_single_var_4D.py` - same as above for simulations of single molecule in vacuum
`analyze_sim_vol_hvap.py` - analyze simulation trajectories to extract molar volumes, HVap, associated uncertainities and state coordinates
`write_lsf_sims_4D.py` - reads `4D_test_new_ref_params_rand_sample.csv` to distribute bulk and vacuum simulations at specified state points

# GP and sampling stuff
the ipython notebook. We need to go through that soon. Get familiar with the above first.
