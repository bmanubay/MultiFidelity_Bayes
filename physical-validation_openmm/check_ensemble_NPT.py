import physical_validation as pv
import pandas as pd


file_path_low = 'cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080_wHConstraints_1fs.csv'
file_path_high = 'cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080_wHConstraints_80.6bar_1fs.csv'

MMcyc = 84.164 #g/mol

vol_conv = 250.*((6.02214e23)**-1)*1.e21

system = pv.data.SystemData(
    natoms=250*18,
    nconstraints=250*18,
    ndof_reduction_tra=3,
    ndof_reduction_rot=0
)

units = pv.data.UnitData(
    kb=8.314462435405199e-3,
    energy_str='kJ/mol',
    energy_conversion=1.0,
    length_str='nm',
    length_conversion=1.0,
    volume_str='cm^3/mol',
    volume_conversion=vol_conv,
    pressure_str='bar',
    pressure_conversion=1.0,
    time_str='fs',
    time_conversion=1.0,
    temperature_str='K',
    temperature_conversion=1.0
)

ensemble_low = pv.data.EnsembleData(
    ensemble='NPT',
    natoms=250*18,
    pressure=1.01,
    temperature=293.15
)

ensemble_high = pv.data.EnsembleData(
    ensemble='NPT',
    natoms=250*18,
    pressure=80.6,
    temperature=293.15
)

low_df = pd.read_csv(file_path_low,sep=',')
high_df = pd.read_csv(file_path_high,sep=',')

low_vol = (low_df["Density (g/mL)"]**(-1))*MMcyc
high_vol = (high_df["Density (g/mL)"]**(-1))*MMcyc

obs_low = pv.data.ObservableData(volume = low_vol)
obs_high = pv.data.ObservableData(volume = high_vol)

res_low = pv.data.SimulationData(
    units=units, ensemble=ensemble_low,
    system=system, observables=obs_low
)

res_high = pv.data.SimulationData(
    units=units, ensemble=ensemble_high,
    system=system, observables=obs_high
)

print('\n## Validating ensemble with different pressures')
quantiles = pv.ensemble.check(res_low, res_high,
                              screen=False,verbosity=3)

