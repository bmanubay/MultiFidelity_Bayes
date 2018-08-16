import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sys import exit
from sys import argv
from pdb import set_trace
import netCDF4 as nc
import mdtraj as md
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import pandas as pd
import pymbar as mb
from pymbar import timeseries
from collections import OrderedDict
from smarty import *
from openforcefield.typing.engines.smirnoff import *
from openforcefield.utils import get_data_filename, generateTopologyFromOEMol, read_molecules
import openmmtools.integrators as ommtoolsints
import mdtraj as md
from itertools import product
import pickle
import re
#-------------------------------------------------
def read_traj(ncfiles,indkeep=0):
    """
    Take multiple .nc files and read in coordinates in order to re-valuate energies based on parameter changes
    Parameters
    -----------
    ncfiles - a list of trajectories in netcdf format
    Returns
    ----------
    data - all of the data contained in the netcdf file
    xyzn - the coordinates from the netcdf in nm
    """

    data = nc.Dataset(ncfiles)
    
    xyz = data.variables['coordinates']
    
    xyzn = Quantity(xyz[indkeep:-1], angstroms)   
    
    lens = data.variables['cell_lengths']
    lensn = Quantity(lens[indkeep:-1], angstroms)

    angs = data.variables['cell_angles']
    angsn = Quantity(angs[indkeep:-1], degrees)

    return data, xyzn, lensn, angsn
#------------------------------------------------------------------
def read_traj_vac(ncfiles,indkeep=0):

    data = nc.Dataset(ncfiles)

    xyz = data.variables['coordinates']
    xyzn = Quantity(xyz[indkeep:-1], angstroms)

    return data, xyzn
#------------------------------------------------------------------
def get_energy(system, positions, vecs):
    """
    Return the potential energy.
    Parameters
    ----------
    system : simtk.openmm.System
        The system to check
    positions : simtk.unit.Quantity of dimension (natoms,3) with units of length
        The positions to use
    vecs : simtk.unit.Quantity of dimension 3 with unit of length
        Box vectors to use 
    Returns
    ---------
    energy
    """
    
    integrator = ommtoolsints.LangevinIntegrator(293.15 * kelvin, 1./picoseconds, 1. * femtoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
    #platform = mm.Platform.getPlatformByName('CUDA')
    #properties = {"CudaPrecision": "mixed","DeterministicForces": "true" }

    #context = mm.Context(system, integrator, platform, properties)
    context = mm.Context(system, integrator, platform)

    context.setPeriodicBoxVectors(*vecs*angstroms)
    context.setPositions(positions)#*angstroms)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() 
    return energy
#------------------------------------------------------------------
def get_energy_vac(system, positions):
    """
    Return the potential energy.
    Parameters
    ----------
    system : simtk.openmm.System
        The system to check
    positions : simtk.unit.Quantity of dimension (natoms,3) with units of length
        The positions to use
    Returns
    ---------
    energy
    """

    integrator = ommtoolsints.LangevinIntegrator(293.15 * kelvin, 1./picoseconds, 0.8 * femtoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
    #platform = mm.Platform.getPlatformByName('CUDA')
    #properties = {"CudaPrecision": "mixed","DeterministicForces": "true" }

    #context = mm.Context(system, integrator, platform, properties)
    context = mm.Context(system, integrator, platform)
 
    context.setPositions(positions)#*angstroms)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    return energy

#---------------------------------------------------
def new_param_energy(coords, params, topology, vecs, T, P=1.01,NPT=False,V=None,P_conv=1.e5,V_conv=1.e-6,Ener_conv=1.e-3,N_part=250.):
    """
    Return potential energies associated with specified parameter perturbations.
    Parameters
    ----------
    coords: coordinates from the simulation(s) ran on the given molecule
    params:  arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :
             params = {'AlkEthOH_c1143':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...}}
    P: Pressure of the system. By default set to 1.01 bar.
    T: Temperature of the system. By default set to 300 K.
    Returns
    -------
    E_kn: a kxN matrix of the dimensional energies associated with the forcfield parameters used as input
    u_kn: a kxN matrix of the dimensionless energies associated with the forcfield parameters used as input
    """

    #-------------------
    # CONSTANTS
    #-------------------
    kB = 0.0083145  #Boltzmann constant (Gas constant) in kJ/(mol*K)
    #beta = 1/(kB*T)

    #-------------------
    # PARAMETERS
    #-------------------
    params = params

    mol2files = []
    for i in params:
        mol2files.append('monomers/'+i.rsplit(' ',1)[0]+'.mol2')

    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    mols = []
    mol = oechem.OEMol()
    for mol2file in mol2files:
        ifs = oechem.oemolistream(mol2file)
        ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
        mol = oechem.OEGraphMol()
        while oechem.OEReadMolecule(ifs, mol):
            oechem.OETriposAtomNames(mol)
            mols.append(oechem.OEGraphMol(mol))
    K = len(params['cyclohexane'].keys())
    
    # Load forcefield file
    ffxml = 'smirnoff99Frosst.ffxml'#
    #print('The forcefield being used is smirnoff99Frosst_with_AllConstraints.ffxml')
    #ffxml = get_data_filename('forcefield/smirnoff99Frosst.ffxml')
    print('The forcefield being used is smirnoff99Frosst.ffxml')

    ff = ForceField(ffxml)

    # Generate a topology
    top = topology#generateTopologyFromOEMol(mol)

    #-----------------
    # MAIN
    #-----------------

    # Calculate energies

    E_kn = np.zeros([K,len(coords)],np.float64)
    u_kn = np.zeros([K,len(coords)],np.float64)
    for i,j in enumerate(params):
        AlkEthOH_id = j
        for k,l in enumerate(params[AlkEthOH_id]):
            print("Anotha one")
            for m,n in enumerate(params[AlkEthOH_id][l]):
                #print(n)
                newparams = ff.getParameter(smirks=n[0])
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0])
                system = ff.createSystem(top,mols,nonbondedMethod=PME,nonbondedCutoff=1.125*nanometers,ewaldErrorTolerance=1.e-5)
                #barostat = MonteCarloBarostat(P*bar, T*kelvin, 10)
                #system.addForce(barostat)
                #print(ff.getParameter(smirks=n[0]))
            for o,p in enumerate(coords):
                barostat = MonteCarloBarostat(P*bar, T[o]*kelvin, 10)
                system.addForce(barostat)
                #print(ff.getParameter(smirks=n[0]))
                e = get_energy(system,p,vecs[o])
                #print(e)
                beta = 1/(kB*T[o])
                #print(e) 
                if not NPT:
                    #print("NVT")
                    E_kn[k,o] = e._value
                    u_kn[k,o] = e._value*beta
                else:
                    #print("NPT")
                    E_kn[k,o] = e._value + P*P_conv*V[o]*V_conv*Ener_conv*N_part
                    u_kn[k,o] = (e._value + P*P_conv*V[o]*V_conv*Ener_conv*N_part)*beta
                    #print(E_kn)
    return E_kn,u_kn

#---------------------------------------------------------------------------
def new_param_energy_vac(coords, params, T=293.15):
    """
    Return potential energies associated with specified parameter perturbations.
    Parameters
    ----------
    coords: coordinates from the simulation(s) ran on the given molecule
    params:  arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :
             params = {'AlkEthOH_c1143':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...}}
    T: Temperature of the system. By default set to 300 K.
    Returns
    -------
    E_kn: a kxN matrix of the dimensional energies associated with the forcfield parameters used as input
    u_kn: a kxN matrix of the dimensionless energies associated with the forcfield parameters used as input
    """

    #-------------------
    # CONSTANTS
    #-------------------
    kB = 0.0083145  #Boltzmann constant (Gas constant) in kJ/(mol*K)
    #beta = 1/(kB*T)

    #-------------------
    # PARAMETERS
    #-------------------
    params = params
    
    # Determine number of states we wish to estimate potential energies for
    mols = []
    for i in params:
        mols.append(i)
    mol = 'monomers/'+mols[0]+'.mol2'
    K = len(params[mols[0]].keys())


    #-------------
    # SYSTEM SETUP
    #-------------
    verbose = False # suppress echos from OEtoolkit functions
    ifs = oechem.oemolistream(mol)
    mol = oechem.OEMol()
    # This uses parm@frosst atom types, so make sure to use the forcefield-flavor reader
    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
    oechem.OEReadMolecule(ifs, mol )
    # Perceive tripos types
    oechem.OETriposAtomNames(mol)

    # Load forcefield file
    #ffxml = 'smirnoff99Frosst_with_AllConstraints.ffxml'#
    #print('The forcefield being used is smirnoff99Frosst_with_AllConstraints.ffxml')
    ffxml = get_data_filename('forcefield/smirnoff99Frosst.ffxml')
    print('The forcefield being used is smirnoff99Frosst.ffxml')

    ff = ForceField(ffxml)

    # Generate a topology
    topology = generateTopologyFromOEMol(mol)

    #-----------------
    # MAIN
    #-----------------

    # Calculate energies

    E_kn = np.zeros([K,len(coords)],np.float64)
    u_kn = np.zeros([K,len(coords)],np.float64)
    for i,j in enumerate(params):
        AlkEthOH_id = j
        for k,l in enumerate(params[AlkEthOH_id]):
            print("Anotha one")
            for m,n in enumerate(params[AlkEthOH_id][l]):
                newparams = ff.getParameter(smirks=n[0]) 
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0])
                system = ff.createSystem(topology, [mol])
            #print(newparams)
            for o,p in enumerate(coords):
                #print(ff.getParameter(smirks='[#6X4:1]'))
                #print(ff.getParameter(smirks='[#1:1]-[#6X4]'))
                beta = 1/(kB*T[o])
                e = get_energy_vac(system,p)
                #print(e)
                E_kn[k,o] = e._value
                u_kn[k,o] = e._value*beta


    return E_kn,u_kn

#-------------------------------------------------------------------------
############################################################
################CONSTANTS AND PARAMETERS####################
############################################################
kB = 0.0083145 #Boltzmann constant (kJ/mol/K)
N_Av = 6.02214085774e23 #particles per mole
N_part = 250. #particles of cyclohexane in box
MMcyc = 84.15948 #g/mol
############################################################

############################################################
###########FILENAMES AND DATA EXTRACTION####################
############################################################
# We define here what data will be used for the reference states in the MBAR calculation
# The list `files` can be of arbitrary length
# Potential TODO: Separate these files from my other simulations and just glob grab all files in the up-to-date `references`  folder
files = ['cyclohexane_250_[#6X4:1]_epsilon0.123075_rmin_half1.9557_[#1:1]-[#6X4]_epsilon0.017662499999999998_rmin_half1.524175.nc','cyclohexane_250_[#6X4:1]_epsilon0.08086972_rmin_half1.99756248_[#1:1]-[#6X4]_epsilon0.00898037_rmin_half1.47905755.nc']

# Extracting the '[#6X4:1]_epsilon<value>_rmin_half<value>' part of the strings from `files`
file_strings = [i.rsplit('.',1)[0].split('_',2)[2] for i in files]

# Creating the correct filenames for the data I want to use
# Configuration and state data trajectories for the liquid and ideal gasw simulations
file_tups_traj = [['traj_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_250_'+i+'_wNoConstraints_1fsts.nc'] for i in file_strings]
file_tups_traj_vac = [['traj_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_'+i+'_wNoConstraints_vacuum_0.8fsts.nc'] for i in file_strings]

file_tups_sd = [['StateData_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_250_'+i+'_wNoConstraints_1fsts.csv'] for i in file_strings]
file_tups_sd_vac = [['StateData_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_'+i+'_wNoConstraints_vacuum_0.8fsts.csv'] for i in file_strings]

# Create a list of lists for the parameters in the reference data. Regex retains order for search
params = [re.findall(r"[-+]?\d*\.\d+", i) for i in file_strings]

# Create a list of SMIRKS strings that were altered during reference simulations (the ones we'll be changing for MBAR) 
file_strings_sep = [i.rsplit('.',1)[0].rsplit('_') for i in files]
smirks_no_filter = [re.findall(r'\[(.*)\]', i) for j in file_strings_sep for i in j]

# Remove empty lists
smirks_no_filter = [x for x in smirks_no_filter if x != []]

# Compress redundant dimensions
smirks_no_filter = list( np.squeeze(smirks_no_filter))
smirks_no_filter = sorted(smirks_no_filter, key=len)

# Add brackets back to SMIRKS strings		
#smirks = ['['+i+']' for i in smirks_no_filter]
smirks = ['[#6X4:1]', '[#6X4:1]', '[#1:1]-[#6X4]', '[#1:1]-[#6X4]'] #Uhhhhh, needs to be fixed. Hard coded for 4D problem. Maybe figuring out a way to create a binary of the parameter state with a function call would be worthwhile.
# Set up containers for all the data I'm extracting from the trajectories
xyz_orig = [[] for i in file_tups_traj] # coordinates from liquid sims
xyz_orig_vac = [[] for i in file_tups_traj] # coordinates from gas sims
vol_orig = [[] for i in file_tups_traj] # molar volumes from liquid sims
ener_orig = [[] for i in file_tups_sd] # potential energies from liquid sims
ener_orig_vac = [[] for i in file_tups_sd] # potential energies from gas sims
vecs_orig = [[] for i in file_tups_sd] # box vectors from liquid sims
vol_box_orig = [[] for i in file_tups_sd] # box volumes from liquid sims
steps_orig = [[] for i in file_tups_sd]
steps_orig_vac = [[] for i in file_tups_sd] # step number from gas sims (debugging those sims, so looking at traces)
temp_orig = [[] for i in file_tups_sd] # temperatures from liquid sims
temp_orig_vac = [[] for i in file_tups_sd] # temperatures from gas sims

# Define number of burnin samples for the liquid and gas sims (could set up better way to do this with equilibration detection)
burnin = 7500#8000#9997#1000#1949
burnin_vac = 7500#9997#1000#3949

print('burnin bulk = %s' %(burnin))
print('burnin vac = %s' %(burnin_vac))

print( 'Extracting data from cyclohexane neat liquid configuration trajectories')
for j,i in enumerate(file_tups_traj):
    for ii in i:            
        try:
            print(ii)
            data, xyz, lens, angs = read_traj(ii,burnin)           
        except IndexError:
            # If the trajectory has fewer than the burnin samples then give warning flag
            print( "The trajectory had fewer than %s frames") %(burnin)
            continue 
            
        for m,n in zip(lens,angs):  
            # Calculate box vectors and then box volumes
            vecs = md.utils.lengths_and_angles_to_box_vectors(float(m[0]._value),float(m[1]._value),float(m[2]._value),float(n[0]._value),float(n[1]._value),float(n[2]._value))        
            vecs_orig[j].append(vecs)
            vol_box_orig[j].append(np.prod(m))

        for pos in xyz:
            xyz_orig[j].append(pos)

print( 'Extracting data from cyclohexane gas configuration trajectories')
for j,i in enumerate(file_tups_traj_vac):
    for ii in i:
        try:
            data_vac, xyz_vac = read_traj_vac(ii,burnin_vac)
        except IndexError:
            # If the trajectory has fewer than the burnin samples then give warning flag
            print( "The trajectory had fewer than %s frames") %(burnin)
            continue

    for pos in xyz_vac:
        xyz_orig_vac[j].append(pos)

print( 'Extracting data from cyclohexane neat liquid state data trajectories')
for j,i in enumerate(file_tups_sd):
    try:
        datasets = [pd.read_csv(ii,sep=',')[burnin:-1] for ii in i]
        merged = pd.concat(datasets)
    except IndexError:
        print( "The state data record had fewer than %s frames") %(burnin)
    
    for e in merged["Potential Energy (kJ/mole)"]:
        ener_orig[j].append(e)##*(N_Av**(-1.))*N_part) #Energy per mol box

    for T in merged["Temperature (K)"]:
        temp_orig[j].append(T)
 
    for dens in merged["Density (g/mL)"]:
        vol_orig[j].append(MMcyc*dens**(-1.)) # converted density to molar volume
    
    for s in merged['#"Step"']:
        steps_orig[j].append(s)

print( 'Extracting data from cyclohexane gas state data trajectories')
for j,i in enumerate(file_tups_sd_vac):
    try:
        datasets = [pd.read_csv(ii,sep=',')[burnin_vac:-1] for ii in i]
        merged = pd.concat(datasets) 
    except IndexError:
        print( "The state data record had fewer than %s frames") %(burnin)
    
    for e in merged["Potential Energy (kJ/mole)"]:
        ener_orig_vac[j].append(e)##*N_Av**(-1)) #Energy per mol box

    for t in merged["Temperature (K)"]:
        temp_orig_vac[j].append(t)

    for s in merged['#"Step"']:
        steps_orig_vac[j].append(s)
  

###################################################################################
####################PARAMETERS OF INTEREST#########################################
###################################################################################
param_types = ['epsilon','rmin_half','epsilon','rmin_half']

for i in params:
    if len(param_types) != len(i):
	    raise ValueError('A value must be specified for each parameter type for each SMIRKS')


###################################################################################
####################SUBSAMPLE DATA#################################################
###################################################################################
# We subsample the data we extracted based on its correlation length

# Define containers for our subsampled trajectories (naming scheme remains the same)
ener_orig_sub = [[] for i in ener_orig]
vol_orig_sub = [[] for i in vol_orig]
vol_box_orig_sub = [[] for i in vol_box_orig]
ener_orig_vac_sub = [[] for i in ener_orig_vac]
xyz_orig_sub = [[] for i in xyz_orig]
xyz_orig_vac_sub = [[] for i in xyz_orig_vac]
vecs_orig_sub = [[] for i in vecs_orig]
steps_orig_sub = [[] for i in steps_orig]
steps_orig_vac_sub = [[] for i in steps_orig_vac]
temp_orig_sub = [[] for i in temp_orig]
temp_orig_vac_sub = [[] for i in temp_orig_sub]
xyz_all_sub = []
xyz_all_vac_sub = []
temp_all_sub = []
temp_all_vac_sub = []
vecs_all_sub = []
# subsample our liquid timeseries based on the correlation lengths of the energies
# all subsampling needs to be identical for each state, if we subsampled each observable individually they would likely be different length
for ii,value in enumerate(ener_orig):
    ts = [value] # Pull out timeseries for given state 
    g = np.zeros(len(ts),np.float64) # set up container for statistical inefficiency (essentially correlation length) calculations 

    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0: # check for timeseries that are just zeros
            g[i] = np.float(1.)
            print( "WARNING FLAG")
        else:
            g[i] = timeseries.statisticalInefficiency(t) # calculate statistical inefficiency of timeseries

    # produce new indices based on statistical inefficiency 
    ind = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]
    inds = ind[0]
    
    # resample observables based on subsampled indices
    print("Sub-sampling")
    ener_sub = [value[j] for j in inds]
    vol_sub = [vol_orig[ii][j] for j in inds]
    vol_box_sub = [vol_box_orig[ii][j] for j in inds]
    xyz_sub = [xyz_orig[ii][j] for j in inds]
    vecs_sub = [vecs_orig[ii][j] for j in inds]
    temp_sub = [temp_orig[ii][j] for j in inds]
    steps_sub = [steps_orig[ii][j] for j in inds]

    ener_orig_sub[ii] = ener_sub
    vol_orig_sub[ii] = vol_sub
    vol_box_orig_sub[ii] = vol_box_sub
    xyz_orig_sub[ii] = xyz_sub
    steps_orig_sub[ii] = steps_sub
    vecs_orig_sub[ii] = vecs_sub
    temp_orig_sub[ii] = temp_sub
    vecs_all_sub.extend(vecs_sub)
    temp_all_sub.extend(temp_sub)
    xyz_all_sub.extend(xyz_sub)

# subsample the gas phase observables 
# since I use separate MBAR objects for the liquid and gas observables, they can be subampled separately
# process is same as above
for ii,value in enumerate(ener_orig_vac):
    ts = [value]
    g = np.zeros(len(ts),np.float64)

    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0:
            g[i] = np.float(1.)
            print( "WARNING FLAG")
        else:
            g[i] = timeseries.statisticalInefficiency(t)

    ind_vac = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]
    inds_vac = ind_vac[0]

    print("Sub-sampling")
    ener_vac_sub = [value[j] for j in inds_vac]
    xyz_vac_sub = [xyz_orig_vac[ii][j] for j in inds_vac]
    steps_vac_sub = [steps_orig_vac[ii][j] for j in inds_vac]
    temp_vac_sub = [temp_orig_vac[ii][j] for j in inds_vac]

    ener_orig_vac_sub[ii] = ener_vac_sub
    xyz_orig_vac_sub[ii] = xyz_vac_sub
    steps_orig_vac_sub[ii] = steps_vac_sub
    temp_orig_vac_sub[ii] = temp_vac_sub
    temp_all_vac_sub.extend(temp_vac_sub)
    xyz_all_vac_sub.extend(xyz_vac_sub)


vol_sub = np.array(vol_orig_sub)
temp_sub = np.array(temp_orig_sub)
temp_vac_sub = np.array(temp_orig_vac_sub)

# rearrange molar volumes so they're in one single array of length sum(N_k_sub) (all states still in order according to the N_k object)
vol_subs = np.zeros([1,sum([len(i) for i in ener_orig_sub])],np.float64)
vol_ind = 0
for i,j in enumerate(vol_sub):
    vol_subs[0,vol_ind:vol_ind+len(j)] = j
    vol_ind+=len(j)

#############################################################################################
#########################CONSTRUCTION OF U_KN MATRIX#########################################
#############################################################################################

# Define new parameter states we wish to evaluate energies at
# Arrays of length 3 produced based on bounds of parameters fed to python command
eps_vals1 = np.array([0.95*float(params[-1][0]),float(params[-1][0]),1.05*float(params[-1][0])])
rmin_vals1 = np.array([0.99*float(params[-1][1]),float(params[-1][1]),1.01*float(params[-1][1])])
eps_vals2 = np.array([0.95*float(params[-1][2]),float(params[-1][2]),1.05*float(params[-1][2])])
rmin_vals2 = np.array([0.99*float(params[-1][3]),float(params[-1][3]),1.01*float(params[-1][3])])

# Turn array of floats into list of strings (the forcefield editing machinery takes string inputs) 
eps_vals1 = [str(a) for a in eps_vals1]
rmin_vals1 = [str(a) for a in rmin_vals1]
eps_vals2 = [str(a) for a in eps_vals2]
rmin_vals2 = [str(a) for a in rmin_vals2]

new_states = list(product(eps_vals1,rmin_vals1,eps_vals2,rmin_vals2))# List of all (eps, rmin_half) combinations 

new_states = list(set(new_states)) # uniquefy list
print(len(new_states))

orig_state = params # the list of tuples from the reference parameter states

# Get rid of a state in 'new_states' if it's already in the 'orig_state' object
for i in new_states:
    if i in orig_state:
        new_states.remove(i)

N_eff_list = []
param_type_list = []
param_val_list = []

"""
state_coords = orig_state
for i in new_states:
     state_coords.append(i)

with open('state_coords_rmin1_stabtesthigh_5ref_randreftest1.pkl', 'wb') as f:
    pickle.dump(state_coords, f)

exit()
"""

# To make this code more distributed process friendly, I run them off of presaved pickles of state arrays and run a subset of the 
# indices across many processes (I make them with this file as well. The triple quote commented out lines above show the way to do this.) 
# `python run_MBAR_arbitrary_dim.py 0 8` runs states 0 through 7 as defined by the pickle file.

pickle_in = open('state_coords_rmin1_stabtesthigh_5ref_randreftest1.pkl','rb')
state_coords = pickle.load(pickle_in)[int(sys.argv[1]):int(sys.argv[2])] #load whatever state coords defined in command line args

# Add the reference states back to the state coordinates being fed to MBAR
for j,i in enumerate(orig_state):
    if i not in state_coords:
        state_coords.insert(j,i)


# load up the packmol box from the liquid simulation for energy re-evaluation
filename = 'packmol_boxes/cyclohexane_250.pdb'
pdb = PDBFile(filename)

# Make new N_k (samples per state) objects based on subsampled energy lengths, but also 
# filling in zeros for the states we're predicting (and don't have samples for)
N_k = np.zeros(len(state_coords),np.int64)
N_k_vac = np.zeros(len(state_coords),np.int64)

for k,i in enumerate(ener_orig_sub): 
    N_k[k] = len(i)
for k,i in enumerate(ener_orig_vac_sub):
    N_k_vac[k] = len(i)

# Begin energy re-evaluations
MBAR_moves = state_coords

print( "Number of MBAR calculations for liquid cyclohexane: %s" %(len(MBAR_moves)))
print( "starting MBAR calculations")
D = OrderedDict()
for i,val in enumerate(MBAR_moves):
    D['State' + ' ' + str(i)] = [[smirks[j],param_types[j],val[j]] for j in range(len(param_types))]
D_mol = {'cyclohexane' : D}
E_kns, u_kns = new_param_energy(xyz_all_sub,D_mol, pdb.topology,vecs_all_sub,T = temp_all_sub,P=1.01,NPT=True,V=vol_subs[0])

print( "Number of MBAR calculations for cyclohexane in vacuum: %s" %(len(MBAR_moves)))
print( "starting MBAR calculations")
E_kns_vac, u_kns_vac = new_param_energy_vac(xyz_all_vac_sub,D_mol,T = temp_all_vac_sub)

# number of states and total number of samples
K,N = np.shape(u_kns)
K_vac,N_vac = np.shape(u_kns_vac)

# We're going to get bootstrapped estimates of variance in order to test the stability of MBAR's asymptotic
# variance estimator. Thus, we'll define the number of bootstrap samples and make some containers for the observables
# we're bootstrapping. 
nBoots_work = 1#2#1000
nBoots_work_vac = 1#2#1000

N_eff_boots = []
u_kn_boots = []
V_boots = []
dV_boots =[]
E_boots = []
dE_boots = []

# Start loop for bootstrap
for n in range(nBoots_work):
    for k in range(len(N_k)):
        if N_k[k] > 0:
            if (n == 0): 
                booti = np.array(range(int(sum(N_k))),int) # first sample should have unaltered indices
                print("First sample does not have indices altered")
            else:
                booti = np.random.randint(int(sum(N_k)), size = int(sum(N_k)))
    print("Bootstrap sample %s of 1000" %(n+1))
    E_kn_boot = E_kns[:,booti]       
    u_kn_boot = u_kns[:,booti]
    vol_sub_boot = vol_subs[:,booti][0]
    u_kn_boots.append(u_kns)
    
    # Initialize MBAR with Newton-Raphson
    # Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
    ########################################################################################  
    if (n==0):
        initial_f_k = None # start from zero 
    else:
        initial_f_k = mbar.f_k # start from the previous final free energies to speed convergence        
    print("begin MBAR")
    mbar = mb.MBAR(u_kn_boot, N_k, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)
    
    O_ij = mbar.computeOverlap()
    print(O_ij) # just printing to stdout, not really worth storing, but somewhat useful for debugging      

    N_eff = mbar.computeEffectiveSampleNumber(verbose=False)
    
    N_eff_boots.append(N_eff)

    (Vol_expect,dVol_expect) = mbar.computeExpectations(vol_sub_boot,state_dependent = False)
    
    V_boots.append(Vol_expect)
    dV_boots.append(dVol_expect)

    (E_expect, dE_expect) = mbar.computeExpectations(E_kn_boot,state_dependent = True)
 
    E_boots.append(E_expect)
    dE_boots.append(dE_expect)
    print("end MBAR")      

# Collect the expectations from the iteration with unaltered indices 
u_kn = u_kn_boots[0]
N_eff = N_eff_boots[0]
Vol_expect = V_boots[0]
dVol_expect = dV_boots[0]
E_expect = E_boots[0]    
dE_expect = dE_boots[0]
     
E_boots_vt = np.vstack(E_boots)
V_boots_vt = np.vstack(V_boots)

# Calculate bootstrapped means and variances
E_bootstrap = [np.mean(E_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Mean of E calculated with bootstrapping
dE_bootstrap = [np.std(E_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Standard error of E from bootstrap
Vol_bootstrap = [np.mean(V_boots_vt[1:,a]) for a in range(np.shape(V_boots_vt)[1])] #Mean of V calculated with bootstrapping
dVol_bootstrap = [np.std(V_boots_vt[1:,a]) for a in range(np.shape(V_boots_vt)[1])] #Standard error of V from bootstrap   

# repeat above for the gas observables   
N_eff_vac_boots = []
u_kn_vac_boots = []
E_vac_boots = []
dE_vac_boots = []
for n in range(nBoots_work_vac):
    for k in range(len(N_k_vac)):
        if N_k_vac[k] > 0:
            if (n == 0):
                booti = np.array(range(int(sum(N_k_vac))),int)
                print("First sample does not have indices altered")
            else:
                booti = np.random.randint(int(sum(N_k_vac)), size = int(sum(N_k_vac)))
    print("Bootstrap sample %s of 1000" %(n+1))
    E_kn_vac = E_kns_vac[:,booti]
    u_kn_vac = u_kns_vac[:,booti]
        
    u_kn_vac_boots.append(u_kn_vac) 

    # Initialize MBAR with Newton-Raphson
    # Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
    ########################################################################################
    if (n==0):
        initial_f_k = None # start from zero
    else:
        initial_f_k = mbar_vac.f_k # start from the previous final free energies to speed convergence

    mbar_vac = mb.MBAR(u_kn_vac, N_k_vac, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)

    N_eff_vac = mbar_vac.computeEffectiveSampleNumber(verbose=False)
        
    N_eff_vac_boots.append(N_eff_vac)        

    (E_vac_expect, dE_vac_expect) = mbar_vac.computeExpectations(E_kn_vac,state_dependent = True)

    E_vac_boots.append(E_vac_expect)
    dE_vac_boots.append(dE_vac_expect)

u_kn_vac = u_kn_vac_boots[0]
N_eff_vac = N_eff_vac_boots[0]
E_vac_expect = E_vac_boots[0]
dE_vac_expect = dE_vac_boots[0]
    
E_vac_boots_vt = np.vstack(E_vac_boots) 

E_vac_bootstrap = [np.mean(E_vac_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Mean of E calculated with bootstrapping
dE_vac_bootstrap = [np.std(E_vac_boots_vt[1:,a]) for a in range(np.shape(E_boots_vt)[1])] #Standard error of E from bootstrap
 

#######################################################################################
# Calculate heat of vaporization
#######################################################################################
Hvap_expect = [((ener_vac - (1/250.)*ener) + 101000.*1.e-3*(0.024465 - v*1.e-6)) for ener_vac,ener,v in zip(E_vac_expect,E_expect,Vol_expect)]
Hvap_bootstrap = [((ener_vac - (1/250.)*ener) + 101000.*1.e-3*(0.024465 - v*1.e-6)) for ener_vac,ener,v in zip(E_vac_bootstrap,E_bootstrap,Vol_bootstrap)]
dHvap_expect = [np.sqrt(dener_vac**2 + ((1/250.)*dener)**2 + (101000.*1.e-3*1.e-6*dv)**2) for dener_vac,dener,dv in zip(dE_vac_expect,dE_expect,dVol_expect)]  
dHvap_bootstrap = [np.sqrt(dener_vac**2 + ((1/250.)*dener)**2 + (101000.*1.e-3*1.e-6*dv)**2) for dener_vac,dener,dv in zip(dE_vac_bootstrap,dE_bootstrap,dVol_bootstrap)]

print(dHvap_expect)
print(dHvap_bootstrap)
print(Vol_expect,Hvap_expect,dVol_expect,dVol_bootstrap,dHvap_expect,dHvap_bootstrap)    

# Store and save data as ';' delimited csv
df = pd.DataFrame(
                      {'param_value': MBAR_moves,
                       'Vol_expect (mL/mol)': Vol_expect,
                       'dVol_expect (mL/mol)': dVol_expect,
                       'Hvap_expect (kJ/mol)': Hvap_expect,
                       'dHvap_expect (kJ/mol)': dHvap_expect,
                       'E_vac_expect (kJ/mol)': E_vac_expect,
                       'dE_vac_expect (kJ/mol)': dE_vac_expect,
                       'E_expect (kJ/mol)': E_expect,
                       'dE_expect (kJ/mol)': dE_expect,
                       'Vol_bootstrap (mL/mol)': Vol_bootstrap,
                       'dVol_bootstrap (mL/mol)': dVol_bootstrap,
                       'Hvap_bootstrap (kJ/mol)': Hvap_bootstrap,
                       'dHvap_bootstrap (kJ/mol)': dHvap_bootstrap,
                       'N_eff': N_eff
                      })

df.to_csv('StateData_cychex_neat/Lang_2_baro10step_pme1e-5/MBAR_estimates/MBAR_estimates_[6X4:1]_[#1:1]-[#6X4]_eps2.5rmin0.5statecoords'+argv[1]+'-'+argv[2]+'baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_5thiter_5ref_stabtesthigh1_Tempseries_8-13.csv',sep=';')

