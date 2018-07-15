import netCDF4 as nc
import pandas as pd

#files = nc.glob('MBAR_estimates_*_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_stabilitytest17thand18threfsof18_epshighrminhigh_Tempseries_6-5.csv')
files = nc.glob('*7-12.csv')

epsC_values = []
rminC_values = []
epsHC_values = []
rminHC_values = []
Vol_expect = []
dVol_expect = []
Hvap_expect = []
dHvap_expect = []
Vol_boot = []
dVol_boot = []
Hvap_boot = []
dHvap_boot = []
N_eff = []
E_vac_expect = []
dE_vac_expect = []
E_expect = []
dE_expect = []

for i in files:
    df = pd.read_csv(i,sep=';')
    print(i,df.columns)
    new_cols = ['eps_vals_C', 'rmin_vals_C', 'eps_vals_H-C', 'rmin_vals_H-C']
    df[new_cols] = df['param_value'].str[1:-1].str.split(',', expand=True).astype(str)

    df['eps_vals_C'] = df['eps_vals_C'].apply(lambda x: x.replace("'",""))
    df['rmin_vals_C'] = df['rmin_vals_C'].apply(lambda x: x.replace("'",""))
    df['eps_vals_H-C'] = df['eps_vals_H-C'].apply(lambda x: x.replace("'",""))
    df['rmin_vals_H-C'] = df['rmin_vals_H-C'].apply(lambda x: x.replace("'",""))

    df['eps_vals_C'] = df['eps_vals_C'].apply(lambda x: float(x))
    df['rmin_vals_C'] = df['rmin_vals_C'].apply(lambda x: float(x))
    df['eps_vals_H-C'] = df['eps_vals_H-C'].apply(lambda x: float(x))
    df['rmin_vals_H-C'] = df['rmin_vals_H-C'].apply(lambda x: float(x))

    epsC_temp = df['eps_vals_C'].values.tolist()
    rminC_temp = df['rmin_vals_C'].values.tolist()
    epsHC_temp = df['eps_vals_H-C'].values.tolist()
    rminHC_temp = df['rmin_vals_H-C'].values.tolist()
    Vol_temp = df['Vol_expect (mL/mol)'].values.tolist()
    dVol_temp = df['dVol_expect (mL/mol)'].values.tolist()
    Hvap_temp = df['Hvap_expect (kJ/mol)'].values.tolist()
    dHvap_temp = df['dHvap_expect (kJ/mol)'].values.tolist()
    Vol_boot_temp = df['Vol_bootstrap (mL/mol)'].values.tolist()
    dVol_boot_temp = df['dVol_bootstrap (mL/mol)'].values.tolist()
    Hvap_boot_temp = df['Hvap_bootstrap (kJ/mol)'].values.tolist()
    dHvap_boot_temp = df['dHvap_bootstrap (kJ/mol)'].values.tolist()
    Neff_temp = df.N_eff.values.tolist()
    E_vac_temp = df['E_vac_expect (kJ/mol)'].values.tolist()
    dE_vac_temp = df['dE_vac_expect (kJ/mol)'].values.tolist()
    E_temp = df['E_expect (kJ/mol)'].values.tolist()
    dE_temp = df['dE_expect (kJ/mol)'].values.tolist()

    for i in epsC_temp:
        epsC_values.append(i)
    for i in rminC_temp:
        rminC_values.append(i)
    for i in epsHC_temp:
        epsHC_values.append(i)
    for i in rminHC_temp:
        rminHC_values.append(i)
    for i in Vol_temp:
        Vol_expect.append(i)
    for i in dVol_temp:
        dVol_expect.append(i)
    for i in Hvap_temp:
        Hvap_expect.append(i)
    for i in dHvap_temp:
        dHvap_expect.append(i)
    for i in Vol_boot_temp:
        Vol_boot.append(i)
    for i in dVol_boot_temp:
        dVol_boot.append(i)
    for i in Hvap_boot_temp:
        Hvap_boot.append(i)
    for i in dHvap_boot_temp:
        dHvap_boot.append(i)
    for i in Neff_temp:
        N_eff.append(i)
    for i in E_vac_temp:
        E_vac_expect.append(i)
    for i in dE_vac_temp:
        dE_vac_expect.append(i)
    for i in E_temp:
        E_expect.append(i)
    for i in dE_temp:
        dE_expect.append(i)

#print(len(eps_values),len(rmin_values),len(Hvap_expect),len(dVol_boot),len(N_eff))


df2 = pd.DataFrame(
                  {'epsilon values C': epsC_values,
                   'rmin_half values C': rminC_values,
                   'epsilon values H-C': epsHC_values,
                   'rmin_half values H-C': rminHC_values,
                   'Vol_expect (mL/mol)': Vol_expect,
                   'dVol_expect (mL/mol)': dVol_expect,
                   'Hvap_expect (kJ/mol)': Hvap_expect,
                   'dHvap_expect (kJ/mol)': dHvap_expect,
                   'Vol_bootstrap (mL/mol)': Vol_boot,
                   'dVol_bootstrap (mL/mol)': dVol_boot,
                   'Hvap_bootstrap (kJ/mol)': Hvap_boot,
                   'dHvap_bootstrap (kJ/mol)': dHvap_boot,
                   'E_vac_expect (kJ/mol)': E_vac_expect,
                   'dE_vac_expect (kJ/mol)': dE_vac_expect,
                   'E_expect (kJ/mol)': E_expect,
                   'dE_expect (kJ/mol)': dE_expect,
                   'N_eff': N_eff
                  })

df2 = df2.drop_duplicates()

#df2.to_csv('merged/MBAR_estimates_[6X4:1]_eps_0.1022-0.1157_rmin_half_1.8870-1.9260_total_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_stabilitytest4threfof4_epshighrminhigh_Tempseries_5-15.csv',sep=';')
#df2.to_csv('merged/MBAR_estimates_[6X4:1]_eps_0.125-0.15_rmin_half_2.35-2.5_total_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_stabilitytest1ref_epshighrminhigh_Tempseries_5-9.csv',sep=';')
#df2.to_csv('merged/MBAR_estimates_[6X4:1]_eps_0.113-0.133_rmin_half_1.666-1.736_total_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_stabilitytest17thand18threfsof18_epshighrminhigh_Tempseries_6-5.csv',sep=';')
df2.to_csv('merged/4D_MBAR_estimates_[6X4:1]_[#1:1]-[#6X4]_2.5eps0.5rmin_justnativeref_7-12.csv',sep=';')
