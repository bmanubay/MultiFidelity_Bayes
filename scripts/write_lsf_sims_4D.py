import pandas as pd
import os
import glob

sim_states = '4D_sim_params.csv'

df = pd.read_csv(sim_states, sep=';')

#states = df['state_coords'].tolist()

eps1 = df['C_eps'].tolist()
rmin_half1 = df['C_rmin'].tolist()
eps2 = df['H-C_eps'].tolist()
rmin_half2 = df['H-C_rmin'].tolist()

fin = 'cychex_vanilla_sim_blank.lsf'
filenames = []
for ind,j in enumerate(eps1):
    in_file = open(fin)
    out_file_name = 'cychex_sim_eps1'+str(eps1[ind])+'_rmin_half1'+str(rmin_half1[ind])+'_eps2'+str(eps2[ind])+'_rmin_half2'+str(rmin_half2[ind])+'.lsf'
    filenames.append(out_file_name)
    out_file = open(out_file_name,'w')
    for line in in_file:
       out_file.write(line)
    out_file.write("python run_molecule_4D.py cyclohexane C1CCCCC1 250 [#6X4:1] epsilon rmin_half "+str(eps1[ind])+" "+str(rmin_half1[ind])+" [#1:1]-[#6X4] "+str(eps2[ind])+" "+str(rmin_half2[ind])+" & \n")
    out_file.write("python run_molecule_single_var_4D.py cyclohexane [#6X4:1] epsilon rmin_half "+str(eps1[ind])+" "+str(rmin_half1[ind])+" [#1:1]-[#6X4] "+str(eps2[ind])+" "+str(rmin_half2[ind])+" & \n") 
    out_file.close()
  
#scripts = glob.glob('cychex_sim_eps0.1*.lsf')
#print(scripts)
for i in filenames:
    os.system('bsub < ' + i)
