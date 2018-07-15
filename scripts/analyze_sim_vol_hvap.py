import glob
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
import mdtraj as md
from itertools import product
import pickle
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
    xyzn - the coordinates from the netcdf in angstroms
    """

    data = nc.Dataset(ncfiles)
    
    xyz = data.variables['coordinates']
    xyzn = np.float64(xyz[indkeep:-2])#Quantity(xyz[indkeep:-2].data, angstroms)
        
    lens = data.variables['cell_lengths']
    lensn = np.float64(lens[indkeep:-2])#Quantity(lens[indkeep:-2], angstroms)

    angs = data.variables['cell_angles']
    angsn = np.float64(angs[indkeep:-2])#Quantity(lens[indkeep:-2], angstroms)
    
    return data, xyzn, lensn, angsn
#------------------------------------------------------------------
def read_traj_vac(ncfiles,indkeep=0):

    data = nc.Dataset(ncfiles)

    xyz = data.variables['coordinates']
    xyzn = np.float64(xyz[indkeep:-2])#Quantity(xyz[indkeep:-2].data, angstroms)

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
   
    integrator = mm.VerletIntegrator(1.0 * femtoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
    
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions*0.1)
    vecs = [0.1*a for a in vecs]
    vecs = np.array(vecs)
    #print vecs 
    context.setPeriodicBoxVectors(*vecs)
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

    integrator = mm.VerletIntegrator(1.0 * femtoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
   
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions*0.1)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    #print energy
    return energy

#---------------------------------------------------
def new_param_energy(coords, params, topology, vecs, P=1.01, T=300.):
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
    beta = 1/(kB*T)

    #-------------------
    # PARAMETERS
    #-------------------
    params = params

    # Determine number of states we wish to estimate potential energies for
    mol2files = []
    for i in params:
        mol2files.append('monomers/'+i.rsplit(' ',1)[0]+'.mol2')
    #    mol2files.append('monomers/'+i.rsplit(' ',1)[1]+'.mol2')
    #print mol2files
    #mol = 'Mol2_files/'+mols[0]+'.mol2'
    #K = len(params[mols[0]].keys())


    #if np.shape(params) != np.shape(N_k): raise "K_k and N_k must have same dimensions"


    # Determine max number of samples to be drawn from any state

    #mol2files = ['monomers/'+sys.argv[1]+'.mol2','monomers/'+sys.argv[4]+'.mol2']

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
    ffxml = get_data_filename('forcefield/smirnoff99Frosst.ffxml')
    ff = ForceField(ffxml)

    # Generate a topology
    #from smarty.forcefield import generateTopologyFromOEMol
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
            for m,n in enumerate(params[AlkEthOH_id][l]):
                newparams = ff.getParameter(smirks=n[0])
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0])
                system = ff.createSystem(top,mols,nonbondedMethod=PME,nonbondedCutoff=12.*angstroms)
                barostat = MonteCarloBarostat(P*bar, T*kelvin, 25)
                system.addForce(barostat)
            
                #print system
                
            for o,p in enumerate(coords):
                e = get_energy(system,p,vecs[o])
                #print o,e
                E_kn[k,o] = e._value
                u_kn[k,o] = e._value*beta
                

    return E_kn,u_kn

def new_param_energy_vac(coords, params, T=300.):
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
    beta = 1/(kB*T)

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


    #if np.shape(params) != np.shape(N_k): raise "K_k and N_k must have same dimensions"


    # Determine max number of samples to be drawn from any state

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
    ffxml = get_data_filename('forcefield/smirnoff99Frosst.ffxml')
    ff = ForceField(ffxml)

    # Generate a topology
    #from smarty.forcefield import generateTopologyFromOEMol
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
            for m,n in enumerate(params[AlkEthOH_id][l]):
                newparams = ff.getParameter(smirks=n[0])
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0])
                system = ff.createSystem(topology, [mol])
            for o,p in enumerate(coords):
                e = get_energy_vac(system,p)
                E_kn[k,o] = e._value
                u_kn[k,o] = e._value*beta


    return E_kn,u_kn

#----------------------------------------------------
def subsampletimeseries(timeser,xyzn,N_k):
    """
    Return a subsampled timeseries based on statistical inefficiency calculations.
    Parameters
    ----------
    timeser: the timeseries to be subsampled
    xyzn: the coordinates associated with each frame of the timeseries to be subsampled
    N_k: original # of samples in each timeseries

    Returns
    ---------
    N_k_sub: new number of samples per timeseries
    ts_sub: the subsampled timeseries
    xyz_sub: the subsampled configuration series
    """
    # Make a copy of the timeseries and make sure is numpy array of floats
    ts = timeser
    xyz = xyzn

    # initialize array of statistical inefficiencies
    g = np.zeros(len(ts),np.float64)


    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0:
            g[i] = np.float(1.)
            print( "WARNING FLAG")
        else:
            g[i] = 2.*timeseries.statisticalInefficiency(t)

    N_k_sub = np.array([len(timeseries.subsampleCorrelatedData(t,g=b)) for t, b in zip(ts,g)])
    ind = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]

    if (N_k_sub == N_k).all():
        ts_sub = ts
        xyz_sub = xyz
        print( "No sub-sampling occurred")
    else:
        print( "Sub-sampling...")
        ts_sub = np.array([t[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)])
        #for c in xyz:
        #    xyz_sub = [c[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)]
        for i,j in enumerate(xyz):
            xyz_sub = [j[ii] for ii in ind[i]]

    return ts_sub, N_k_sub, xyz_sub, ind
#-------------------------------------------------------------------------

kB = 0.0083145 #Boltzmann constant (kJ/mol/K)
T = 293.15 #Temperature (K)
N_Av = 6.02214085774e23 #particles per mole
N_part = 250. #particles of cyclohexane in box

files = glob.glob('cyclohexane_250_*1fsts.csv')
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080_wAllConstraints_2fsts.csv']
file_strings = [i.rsplit('.',1)[0].rsplit('_',1)[0].split('_',2)[2] for i in files]
#print(file_strings)
file_str = set(file_strings)

file_tups_sd = [['cyclohexane_250_'+i+'_1fsts.csv'] for i in file_str]
file_tups_sd_vac = [['cyclohexane_'+i+'_vacuum_0.8fsts.csv'] for i in file_str]

#print(file_tups_sd_vac)
params = [j.rsplit('.',1)[0].rsplit('_') for i in file_tups_sd for j in i]
params = [[i[3][7:],i[5][4:],i[7][7:],i[9][4:]] for i in params]

MMcyc = 84.15948 #g/mol

 
states_sd = [[] for i in file_tups_sd]
ener_orig = [[] for i in file_tups_sd]
ener_orig_vac = [[] for i in file_tups_sd]
vol_orig = [[] for i in file_tups_sd]
burnin = 1000
burnin_vac = 1000
for j,i in enumerate(file_tups_sd): 
    print( i)
    try:
        datasets = [pd.read_csv(ii,sep=',')[burnin:-1] for ii in i]
        merged = pd.concat(datasets)
        #if len(merged["Potential Energy (kJ/mole)"]) < 2000.:
        #    print(i)
        #    print("this on is short")
        #    continue
    except IndexError:
        print( "The state data record had fewer than %s frames") %(burnin)
    for e,dens in zip(merged["Potential Energy (kJ/mole)"],merged["Density (g/mL)"]):
        ener_orig[j].append(e + N_part*101000.*1.e-3*1.e-6*MMcyc*dens**(-1))
        vol_orig[j].append(MMcyc*dens**(-1))
    print(np.mean(merged["Temperature (K)"]))
    states_sd[j].append(i[0].rsplit('.',1)[0])

for j,i in enumerate(file_tups_sd_vac):
    try:
        print(i)
        datasets = [pd.read_csv(ii,sep=',')[burnin_vac:-1] for ii in i]
        merged = pd.concat(datasets)
        #if len(merged["Potential Energy (kJ/mole)"]) < 2000.:
        #    print(i)
        #    print("this on is short")
        #    continue
    except IndexError:
        print( "The state data record had fewer than %s frames") %(burnin)
    for e in merged["Potential Energy (kJ/mole)"]:
        ener_orig_vac[j].append(e)

#print([len(i) for i in vol_orig])
#print([len(i) for i in ener_orig])
#print([len(i) for i in ener_orig_vac])
#print(states_sd)
#print(params)
#print(len(params))
#print(len(vol_orig))
#print([np.mean(j) for j in vol_orig])

state_coord = params
param_types = ['epsilon','rmin_half']
ener_orig_sub = [[] for i in ener_orig]
vol_orig_sub = [[] for i in vol_orig]
ener_orig_vac_sub = [[] for i in ener_orig_vac]
N_k_sub = []
N_k_vac_sub = []
# Re-evaluate energies from original parameter state to subsample and use that indexing for the other energies and MBAR calculations
for ii,value in enumerate(ener_orig): 
    print(file_tups_sd[ii])
    ts = [value]
    g = np.zeros(len(ts),np.float64)

    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0:
            g[i] = np.float(1.)
            print( "WARNING FLAG")
            print(file_tups_sd[ii])
        else:
            g[i] = timeseries.statisticalInefficiency(t)

    N_k_sub.append([len(timeseries.subsampleCorrelatedData(t,g=b)) for t, b in zip(ts,g)][0])
    ind = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]
    inds = ind[0]
    print("Sub-sampling")
    ener_sub = [value[j] for j in inds]
    vol_sub = [vol_orig[ii][j] for j in inds]
    #if (N_k_sub == N_k).all():
    #    ts_sub = ts
    #    xyz_sub = xyz
    #    print( "No sub-sampling occurred")
    #else:
    
    #ts_sub = np.array([t[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)])
    
    ener_orig_sub[ii] = ener_sub
    vol_orig_sub[ii] = vol_sub

for ii,value in enumerate(ener_orig_vac):
    print(file_tups_sd_vac[ii])
    ts = [value]
    g = np.zeros(len(ts),np.float64)

    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0:
            g[i] = np.float(1.)
            print( "WARNING FLAG")
            print(file_tups_sd_vac[ii])
        else:
            g[i] = timeseries.statisticalInefficiency(t)

    N_k_vac_sub.append([len(timeseries.subsampleCorrelatedData(t,g=b)) for t, b in zip(ts,g)][0])
    ind = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]
    inds_vac = ind[0]
    print("Sub-sampling")
    ener_vac_sub = [value[j] for j in inds_vac]
    
    #if (N_k_sub == N_k).all():
    #    ts_sub = ts
    #    xyz_sub = xyz
    #    print( "No sub-sampling occurred")
    #else:

    #ts_sub = np.array([t[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)])

    ener_orig_vac_sub[ii] = ener_vac_sub
#print(len(ener_orig_sub))
ener_orig_sub = np.array(ener_orig_sub)
ener_orig_vac_sub = np.array(ener_orig_vac_sub)
vol_orig_sub = np.array(vol_orig_sub)
#Hvap = (np.mean(ener_orig_vac_sub) - (1/250.)*np.mean(ener_orig_sub)) + 101000.*1.e-3*(0.024465 - np.mean(vol_orig_sub)*1.e-6)

#implement bootstrapping to get variance of Cp estimate
V_boots = []
E_boots = []
ener_orig_sub_boot = [[] for i in ener_orig_sub]
vol_orig_sub_boot = [[] for i in vol_orig_sub]
nBoots_work = 1000
for n in range(nBoots_work):
    for k in range(len(N_k_sub)):
        if N_k_sub[k] > 0:
            if (n == 0):
                booti = np.array(range(N_k_sub[k]),int)
            else:
                booti = np.random.randint(N_k_sub[k], size = N_k_sub[k])
        #print(k)
        #print(booti)
        #set_trace()
        ener_orig_sub_boot[k] = [ener_orig_sub[k][ind] for ind in booti]
        vol_orig_sub_boot[k] = [vol_orig_sub[k][ind] for ind in booti]
    E_bulk_sim = []
    #dCp_bulk_sim = []
    #Cp_vac_sim = []
    #dCp_vac_sim = []
    for i in ener_orig_sub_boot:
        E_mean = np.mean(i)
        #E2_mean = np.mean([(j)**2 for j in i])/(250.**2)
        #fluc = np.array([E2_mean - E_mean**2])#[(ii - E_mean)**2 for ii in i]
        #Cp_bulk_sim.append(fluc[0] / (kB * T**2))#np.mean(fluc) / (kB * T**2))
        E_bulk_sim.append(E_mean)
    #dCp_bulk_sim.append(np.std(fluc) / (kB * T**2))
    E_boots.append(E_bulk_sim)
    V_boots.append([np.mean(j) for j in vol_orig_sub_boot])

E_boots_vert = np.column_stack(E_boots)
V_boots_vert = np.column_stack(V_boots)
E_bulk_sim = E_boots[0]
V_sim = V_boots[0]
dE_bulk_bootstrap = [np.std(i) for i in E_boots_vert]
dV_bootstrap = [np.std(i) for i in V_boots_vert]

E_vac_boots = []
ener_orig_vac_sub_boot = [[] for i in ener_orig_vac_sub]
nBoots_work = 1000
for n in range(nBoots_work):
    for k in range(len(N_k_vac_sub)):
        if N_k_vac_sub[k] > 0:
            if (n == 0):
                booti = np.array(range(N_k_vac_sub[k]),int)
            else:
                booti = np.random.randint(N_k_vac_sub[k], size = N_k_vac_sub[k])
        #print(k)
        #print(booti)
        #set_trace()
        ener_orig_vac_sub_boot[k] = [ener_orig_vac_sub[k][ind] for ind in booti]
    
    E_vac_sim = []
    for i in ener_orig_vac_sub_boot:
        E_mean = np.mean(i)
        #E2_mean = np.mean([j**2 for j in i])
        #fluc = np.array([E2_mean - E_mean**2])#[(ii - E_mean)**2 for ii in i]
        #Cp_vac_sim.append(fluc[0] / (kB * T**2))#np.mean(fluc) / (kB * T**2))
        #dCp_vac_sim.append(np.std(fluc) / (kB * T**2))
        E_vac_sim.append(E_mean)
    E_vac_boots.append(E_vac_sim)
E_vac_boots_vert = np.column_stack(E_vac_boots)
E_vac_sim = E_vac_boots[0]
dE_vac_bootstrap = [np.std(i) for i in E_vac_boots_vert]
   
Vol_sim = V_sim
dVol_sim = dV_bootstrap

Hvap_sim = [((ener_vac - (1/250.)*ener) + 101000.*1.e-3*(0.024465 - v*1.e-6)) for ener_vac,ener,v in zip(E_vac_sim,E_bulk_sim,Vol_sim)]
dHvap_sim = [np.sqrt(dener_vac**2 + ((1/250.)*dener)**2 + (101000.*1.e-3*1.e-6*dv)**2) for dener_vac,dener,dv in zip(dE_vac_bootstrap,dE_bulk_bootstrap,dVol_sim)]

eps1 = [float(i[0]) for i in params]
rmin1 = [float(i[1]) for i in params]
eps2 = [float(i[2]) for i in params]
rmin2 = [float(i[3]) for i in params]

#print(eps)
#print(len(eps))
#print(rmin)
#print(len(rmin))
print(Vol_sim)
print(len(Vol_sim))
#print(dVol_sim)
#print(len(dVol_sim))
print(Hvap_sim)
#print(len(Cp_res_sim))
print(dHvap_sim)
#print(len(dCp_res_sim))

df = pd.DataFrame(
                          {'[#6X4:1]_eps_val': eps1,
                           '[#6X4:1]_rmin_half_val': rmin1,
                           '[1:1]-[#6X4]_eps_val': eps2,
                           '[1:1]-[#6X4]_rmin_half_val': rmin2,
                           'Vol_sim (mL/mol)': Vol_sim,
                           'dVol_sim (mL/mol)': dVol_sim,
                           'Hvap_sim (kJ/mol)': Hvap_sim,
                           'dHvap_sim (kJ/mol)': dHvap_sim,
                           'E_bulk_sim (kJ/mol)': E_bulk_sim,
                           'dE_bulk_sim (kJ/mol)': dE_bulk_bootstrap,
                           'E_vac_sim (kJ/mol)': E_vac_sim,
                           'dE_vac_sim (kJ/mol)': dE_vac_bootstrap
                          })

df.to_csv('cychex_simulated_observables_4D_param_changes.csv',sep=';')
exit()

# Re-evaluate energies from original parameter state to subsample and use that indexing for the other energies and MBAR calculations
for ii,value in enumerate(all_xyz_vac):
    print( "Recomputing energies of cyclohexane in vacuum") #%(len(state_coords))
    print( "starting energy evaluation")
    D = OrderedDict()
    for i,val in enumerate(state_coords):
        D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    D_mol = {'cyclohexane' : D}

    # Return energies from native parameter state to use as timeseries to subsample
    energies_vac, u_vac = new_param_energy_vac(all_xyz_vac[ii], D_mol, T = 293.15)
    #energies_vac_294, u_vac_294 = new_param_energy_vac(all_xyz_vac[ii], D_mol, T = 294.15)
    #energies_vac_292, u_vac_292 = new_param_energy_vac(all_xyz_vac[ii], D_mol, T = 292.15)
    #print np.shape(u_vac)

    ts_vac_subc,N_k_vac_sub,xyz_vac_sub,ind_vac_subs = subsampletimeseries([energies_vac],[all_xyz_vac[ii]],[len(all_xyz_vac[ii])])

######################################################################################### WORKS^^
# Define new parameter states we wish to evaluate energies at

eps_vals = np.linspace(float(argv[1]),float(argv[2]),3)
rmin_vals = np.linspace(float(argv[3]),float(argv[4]),3)
eps_vals = [str(a) for a in eps_vals]
rmin_vals = [str(a) for a in rmin_vals]
new_states = list(product(eps_vals,rmin_vals))

orig_state = ('0.1094','1.9080')

N_eff_list = []
param_type_list = []
param_val_list = []

#new_state = [str(more_array[index+1])]
state_coords = []
state_coords.append(orig_state)
#state_coords.append(new_state)
for i in new_states:
     state_coords.append(i)
print( state_coords)
    #A_expectations = [[] for a in state_coords]
    #A_var_expectations = [[] for a in state_coords]
    #A_var_alt_expectations = [[] for a in state_coords]
    #dA_expectations = [[] for a in state_coords]
    #dA_var_expectations = [[] for a in state_coords]

for ii,value in enumerate(xyz_sub):
    MBAR_moves = state_coords
    print( "Number of MBAR calculations for liquid cyclohexane: %s" %(len(MBAR_moves)))
    print( "starting MBAR calculations")
    D = OrderedDict()
    for i,val in enumerate(MBAR_moves):
        D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    D_mol = {'cyclohexane' : D} 
        
    # Produce the u_kn matrix for MBAR based on the subsampled configurations
    E_kn, u_kn = new_param_energy(xyz_sub[ii], D_mol, pdb.topology, vecs_sub, T = 293.15)
    E_kn_292, u_kn_292 = new_param_energy(xyz_sub[ii], D_mol, pdb.topology, vecs_sub, T = 292.15)
    E_kn_294, u_kn_294 = new_param_energy(xyz_sub[ii], D_mol, pdb.topology, vecs_sub, T = 294.15)
     
    K,N = np.shape(u_kn)
        
    N_k = np.zeros(K)
    N_k[0] = N
    
    #implement bootstrapping to get variance of Cp estimate
    N_eff_boots = []
    u_kn_boots = []
    V_boots = []
    dV_boots =[]
    Cp_boots = []
    dCp_boots = []
    nBoots_work = 200
    for n in range(nBoots_work):
        for k in range(len(N_k_sub)):
            if N_k_sub[k] > 0:
                if (n == 0):
                    booti = np.array(range(N_k_sub[k]),int)
                else:
                    booti = np.random.randint(N_k_sub[k], size = N_k_sub[k])
        
        E_kn_boot = E_kn[:,booti]       
        u_kn_boot = u_kn[:,booti]
   
        u_kn_boots.append(u_kn)
        
        # Initialize MBAR with Newton-Raphson
        # Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
        ########################################################################################  
        initial_f_k = None # start from zero
        mbar = mb.MBAR(u_kn_boot, N_k, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)
          
        N_eff = mbar.computeEffectiveSampleNumber(verbose=True)
        
        N_eff_boots.append(N_eff)
        #N_eff_list.append(N_eff[1])
        #param_type_list.append(param_types)
        #param_val_list.append(MBAR_moves)

        (E_expect, dE_expect) = mbar.computeExpectations(E_kn_boot,state_dependent = True)
        #(E2_expect, dE2_expect) = mbar.computeExpectations(E_kn**2,state_dependent = True)
        (Vol_expect,dVol_expect) = mbar.computeExpectations(vol_sub,state_dependent = False)
        
        # Convert Box volumes to molar volumes
        Vol_expect = Vol_expect*N_Av*1.e-24 / N_part
        dVol_expect = dVol_expect*N_Av*1.e-24 / N_part

        V_boots.append(Vol_expect)
        dV_boots.append(dVol_expect)
          
        (H_expect,dH_expect) = mbar.computeExpectations(H_sub,state_dependent = False)
    
        E_fluc_input = np.array([(E_kn[i][:] - E_expect[i])**2 for i in range(len(E_expect))])
        (E_fluc_expect, dE_fluc_expect) = mbar.computeExpectations(E_fluc_input,state_dependent = True)
        
        C_p_expect_meth2 = E_fluc_expect/(kB * T**2)
        dC_p_expect_meth2 = dE_fluc_expect/(kB * T**2)

        Cp_boots.append(C_p_expect_meth2)
        dCp_boots.append(dC_p_expect_meth2)
        
    u_kn = u_kn_boots[0]
    N_eff = N_eff_boots[0]
    Vol_expect = V_boots[0]
    dVol_expect = dV_boots[0]
    C_p_expect_meth2 = Cp_boots[0]
    dC_p_expect_meth2 = dCp_boots[0]
 
    Cp_boots_vt = np.vstack(Cp_boots)
    V_boots_vt = np.vstack(V_boots)

    C_p_bootstrap = [np.mean(Cp_boots_vt[:,a]) for a in range(np.shape(Cp_boots_vt)[1])] #Mean of Cp calculated with bootstrapping
    dC_p_bootstrap = [np.std(Cp_boots_vt[:,a])/float(len(Cp_boots_vt[:,a])) for a in range(np.shape(Cp_boots_vt)[1])] #Standard error of Cp from bootstrap
    Vol_bootstrap = [np.mean(V_boots_vt[:,a]) for a in range(np.shape(V_boots_vt)[1])] #Mean of Cp calculated with bootstrapping
    dVol_bootstrap = [np.std(V_boots_vt[:,a])/float(len(V_boots_vt[:,a])) for a in range(np.shape(V_boots_vt)[1])] #Standard error of Cp from bootstrap   



    print( "Number of MBAR calculations for cyclohexane in vacuum: %s" %(len(MBAR_moves)))
    print( "starting MBAR calculations")
    D = OrderedDict()
    for i,val in enumerate(MBAR_moves):
        D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    D_mol = {'cyclohexane' : D}

    # Produce the u_kn matrix for MBAR based on the subsampled configurations
    E_kn_vac, u_kn_vac = new_param_energy_vac(xyz_vac_sub[ii], D_mol, T = 293.15)
    E_kn_vac_292, u_kn_vac_292 = new_param_energy_vac(xyz_vac_sub[ii], D_mol, T = 292.15)
    E_kn_vac_294, u_kn_vac_294 = new_param_energy_vac(xyz_vac_sub[ii], D_mol, T = 294.15)

    K_vac,N_vac = np.shape(u_kn_vac)

    N_k_vac = np.zeros(K_vac)
    N_k_vac[0] = N_vac

    N_eff_vac_boots = []
    u_kn_vac_boots = []
    Cp_vac_boots = []
    dCp_vac_boots = []
    nBoots_work = 200
    for n in range(nBoots_work):
        for k in range(len(N_k_vac_sub)):
            if N_k_vac_sub[k] > 0:
                if (n == 0):
                    booti = np.array(range(N_k_vac_sub[k]),int)
                else:
                    booti = np.random.randint(N_k_vac_sub[k], size = N_k_vac_sub[k])

        E_kn_vac = E_kn_vac[:,booti]
        u_kn_vac = u_kn_vac[:,booti]
        
        u_kn_vac_boots.append(u_kn_vac) 
        # Initialize MBAR with Newton-Raphson
        # Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
        ########################################################################################
        initial_f_k = None # start from zero
        mbar_vac = mb.MBAR(u_kn_vac, N_k_vac, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)

        N_eff_vac = mbar_vac.computeEffectiveSampleNumber(verbose=True)
        
        N_eff_vac_boots.append(N_eff_vac)        

        (E_vac_expect, dE_vac_expect) = mbar_vac.computeExpectations(E_kn_vac,state_dependent = True)
    
        E_vac_fluc_input = np.array([(E_kn_vac[i][:] - E_vac_expect[i])**2 for i in range(len(E_vac_expect))])
        (E_vac_fluc_expect, dE_vac_fluc_expect) = mbar_vac.computeExpectations(E_vac_fluc_input,state_dependent = True)

        C_p_vac_expect_meth2 = E_vac_fluc_expect/(kB * T**2)
        dC_p_vac_expect_meth2 = dE_vac_fluc_expect/(kB * T**2)
        
        Cp_vac_boots.append(C_p_vac_expect_meth2)
        dCp_vac_boots.append(C_p_vac_expect_meth2)

    u_kn_vac = u_kn_vac_boots[0]
    N_eff_vac = N_eff_vac_boots[0]
    C_p_vac_expect_meth2 = Cp_vac_boots[0]
    dC_p_vac_expect_meth2 = dCp_vac_boots[0]

    Cp_vac_boots_vt = np.vstack(Cp_vac_boots) 

    C_p_vac_bootstrap = [np.mean(Cp_vac_boots_vt[:,a]) for a in range(np.shape(Cp_boots_vt)[1])] #Mean of Cp calculated with bootstrapping
    dC_p_vac_bootstrap = [np.std(Cp_vac_boots_vt[:,a])/float(len(Cp_boots_vt[:,a])) for a in range(np.shape(Cp_boots_vt)[1])] #Standard error of Cp from bootstrap
    #######################################################################################
    # Calculate residual heat capacity
    #######################################################################################
    C_p_res_expect = [bulk - gas for bulk,gas in zip(C_p_expect_meth2, C_p_vac_expect_meth2)]
    dC_p_res_expect = [np.sqrt(bulk**2 + gas**2) for bulk,gas in zip(dC_p_expect_meth2, dC_p_vac_expect_meth2)]
    
    C_p_res_bootstrap = [bulk - gas for bulk,gas in zip(C_p_bootstrap, C_p_vac_bootstrap)]
    dC_p_res_bootstrap = [np.sqrt(bulk**2 + gas**2) for bulk,gas in zip(dC_p_bootstrap, dC_p_vac_bootstrap)]

    df = pd.DataFrame(
                          {'param_value': MBAR_moves,
                           'Vol_expect (mL/mol)': Vol_expect,
                           'dVol_expect (mL/mol)': dVol_expect,
                           'C_p_res_expect (J/mol/K)': C_p_res_expect,
                           'dC_p_res_expect (J/mol/K)': dC_p_res_expect,
                           'Vol_bootstrap (mL/mol)': Vol_bootstrap,
                           'dVol_bootstrap (mL/mol)': dVol_bootstrap,
                           'C_p_res_bootstrap (J/mol/K)': C_p_res_bootstrap,
                           'dC_p_res_bootstrap (J/mol/K)': dC_p_res_bootstrap,
                           'N_eff': N_eff
                          })
    df.to_csv('MBAR_estimates_[6X4:1]_eps'+argv[1]+'-'+argv[2]+'_rmin'+argv[3]+'-'+argv[4]+'_2.csv',sep=';')
    
    with open('param_states2.pkl', 'wb') as f:
        pickle.dump(MBAR_moves, f)
    with open('u_kn_bulk2.pkl', 'wb') as f:
        pickle.dump(u_kn, f)
    with open('u_kn_vac2.pkl', 'wb') as f:
        pickle.dump(u_kn_vac, f)

"""
files = nc.glob('MBAR_estimates_*2.csv')

eps_values = []
rmin_values = []
Vol_expect = []
dVol_expect = []
C_p_expect = []
dC_p_expect = []
Vol_boot = []
dVol_boot = []
C_p_boot = []
dC_p_boot = []
N_eff = []

for i in files:
    df = pd.read_csv(i,sep=';')
    print(i,df.columns)
    new_cols = ['eps_vals', 'rmin_vals']
    df[new_cols] = df['param_value'].str[1:-1].str.split(',', expand=True).astype(str)
    
    df['eps_vals'] = df.eps_vals.apply(lambda x: x.replace("'",""))
    df['rmin_vals'] = df.rmin_vals.apply(lambda x: x.replace("'",""))
    
    df['eps_vals'] = df.eps_vals.apply(lambda x: float(x))
    df['rmin_vals'] = df.rmin_vals.apply(lambda x: float(x))
 
    eps_temp = df.eps_vals.values.tolist()
    rmin_temp = df.rmin_vals.values.tolist()
    Vol_temp = df['Vol_expect (mL/mol)'].values.tolist()
    dVol_temp = df['dVol_expect (mL/mol)'].values.tolist()
    Cp_temp = df['C_p_res_expect (J/mol/K)'].values.tolist()
    dCp_temp = df['dC_p_res_expect (J/mol/K)'].values.tolist()
    Vol_boot_temp = df['Vol_bootstrap (mL/mol)'].values.tolist()
    dVol_boot_temp = df['dVol_bootstrap (mL/mol)'].values.tolist()
    Cp_boot_temp = df['C_p_res_bootstrap (J/mol/K)'].values.tolist()
    dCp_boot_temp = df['dC_p_res_bootstrap (J/mol/K)'].values.tolist()
    Neff_temp = df.N_eff.values.tolist()

    for i in eps_temp:
        eps_values.append(i)
    for i in rmin_temp:
        rmin_values.append(i)
    for i in Vol_temp:
        Vol_expect.append(i)
    for i in dVol_temp:
        dVol_expect.append(i)
    for i in Cp_temp:
        C_p_expect.append(i)
    for i in dCp_temp:
        dC_p_expect.append(i)
    for i in Vol_boot_temp:
        Vol_boot.append(i)
    for i in dVol_boot_temp:
        dVol_boot.append(i)
    for i in Cp_boot_temp:
        C_p_boot.append(i)
    for i in dCp_boot_temp:
        dC_p_boot.append(i)
    for i in Neff_temp:
        N_eff.append(i)
print(len(eps_values),len(rmin_values),len(C_p_expect),len(dVol_boot),len(N_eff))
df2 = pd.DataFrame(
                  {'epsilon values': eps_values,
                   'rmin_half values': rmin_values,
                   'Vol_expect (mL/mol)': Vol_expect,
                   'dVol_expect (mL/mol)': dVol_expect,
                   'C_p_res_expect (J/mol/K)': C_p_expect,
                   'dC_p_res_expect (J/mol/K)': dC_p_expect,
                   'Vol_bootstrap (mL/mol)': Vol_boot,
                   'dVol_bootstrap (mL/mol)': dVol_boot,
                   'C_p_res_bootstrap (J/mol/K)': C_p_boot,
                   'dC_p_res_bootstrap (J/mol/K)': dC_p_boot,
                   'N_eff': N_eff
                  })
df2.to_csv('MBAR_estimates_[6X4:1]_eps_0.1022-0.1157_rmin_half_1.8870-1.9260_total.csv',sep=';')
"""
"""
files = nc.glob('*_cont.nc')
file_strings = [i.rsplit('_',1)[0] for i in files]
file_str = set(file_strings)

file_tups_traj = [[i+'.nc',i+'_cont.nc'] for i in file_str]

file_tups_sd = [[i+'.csv',i+'_cont.csv'] for i in file_str]
params = [i.rsplit('.',1)[0].rsplit('_') for i in files]
params = [[i[3][7:],i[5][4:]] for i in params]
MMcyc = 0.08416 #kg/mol

states_traj = [[] for i in file_tups_traj]
states_sd = [[] for i in file_tups_sd]
xyz_orig = [[] for i in file_tups_traj]
vol_orig = [[] for i in file_tups_traj]
lens_orig = [[] for i in file_tups_traj]
angs_orig = [[] for i in file_tups_traj]
ener_orig = [[] for i in file_tups_sd]
dens_orig = [[] for i in file_tups_sd]
vecs_orig = [[] for i in file_tups_sd]
burnin = 1800

print 'Analyzing Cyclohexane neat liquid trajectories'
for j,i in enumerate(file_tups_traj):
    #print 'Analyzing trajectory %s of %s'%(j+1,len(file_tups_traj)+1)
    for ii in i:
        try:
            data, xyz, lens, angs = read_traj(ii,burnin)
            #set_trace()
        except IndexError:
            print "The trajectory had fewer than %s frames" %(burnin)
            continue
        for m,n in zip(lens,angs):
            vecs = md.utils.lengths_and_angles_to_box_vectors(float(m[0]),float(m[1]),float(m[2]),float(n[0]),float(n[1]),float(n[2]))
            vecs_orig[j].append(vecs)
        for pos in xyz:
            xyz_orig[j].append(pos)
        for l in lens:
            vol_orig[j].append(np.prod(l)) #A**3
            lens_orig[j].append(l)
        for ang in angs:
            angs_orig[j].append(ang)
    states_traj[j].append(i[0].rsplit('.',1)[0])

print [np.average(i) for i in vol_orig]
#print Vol_expect
ener2_orig = [[] for i in file_tups_sd]
for j,i in enumerate(file_tups_sd):
    try:
        datasets = [pd.read_csv(ii,sep=',')[burnin:] for ii in i]
        merged = pd.concat(datasets)
    except IndexError:
        print "The state data record had fewer than 250 frames"
    for e in merged["Potential Energy (kJ/mole)"]:
        ener_orig[j].append(e)
        ener2_orig[j].append(e**2)
    for dens in merged["Density (g/mL)"]:
        dens_orig[j].append(dens)
    states_sd[j].append(i[0].rsplit('.',1)[0])

#print ener_orig

print [(np.average(j) - np.average(i)**2)/(kB * T**2) for j,i in zip(ener2_orig,ener_orig)]
print [np.average((i - np.average(i))**2)/(kB * T**2) for i in ener_orig]
print C_p_expect_meth1
print C_p_expect_meth2
"""
