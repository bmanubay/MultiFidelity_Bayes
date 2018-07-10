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
    
    integrator = ommtoolsints.LangevinIntegrator(293.15 * kelvin, 1./picoseconds, 2. * femtoseconds)
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

    integrator = ommtoolsints.LangevinIntegrator(293.15 * kelvin, 1./picoseconds, 1.5 * femtoseconds)
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
def new_param_energy(coords, params, topology, vecs, P=1.01, T=293.15,NPT=False,V=None,P_conv=1.e5,V_conv=1.e-6,Ener_conv=1.e-3,N_part=250.):
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
    #ffxml = 'smirnoff99Frosst_with_AllConstraints.ffxml'#
    #print('The forcefield being used is smirnoff99Frosst_with_AllConstraints.ffxml')
    ffxml = get_data_filename('forcefield/smirnoff99Frosst.ffxml')
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
                newparams = ff.getParameter(smirks=n[0])
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0])
                system = ff.createSystem(top,mols,nonbondedMethod=PME,nonbondedCutoff=1.125*nanometers,ewaldErrorTolerance=1.e-5)
                barostat = MonteCarloBarostat(P*bar, T*kelvin, 10)
                system.addForce(barostat)
            for o,p in enumerate(coords):
                e = get_energy(system,p,vecs[o])
               
                if not NPT:
                    E_kn[k,o] = e._value
                    u_kn[k,o] = e._value*beta
                else:
                    E_kn[k,o] = e._value + P*P_conv*V[o]*V_conv*Ener_conv*N_part
                    u_kn[k,o] = (e._value + P*P_conv*V[o]*V_conv*Ener_conv*N_part)*beta
    
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
                e = get_energy_vac(system,p)
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
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080.nc','cyclohexane_250_[#6X4:1]_epsilon0.110623_rmin_half1.8870.nc','cyclohexane_250_[#6X4:1]_epsilon0.114614125174_rmin_half1.88714816576.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.085_rmin_half2.45.nc','cyclohexane_250_[#6X4:1]_epsilon0.0350482467832_rmin_half2.34738763255.nc','cyclohexane_250_[#6X4:1]_epsilon0.1085_rmin_half2.285.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.085_rmin_half2.45.nc','cyclohexane_250_[#6X4:1]_epsilon0.1085_rmin_half2.285.nc','cyclohexane_250_[#6X4:1]_epsilon0.0590978610897_rmin_half2.07011439037.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.1085_rmin_half2.285.nc','cyclohexane_250_[#6X4:1]_epsilon0.0590978610897_rmin_half2.07011439037.nc','cyclohexane_250_[#6X4:1]_epsilon0.109_rmin_half1.9296.nc'] 
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.109_rmin_half1.9296.nc','cyclohexane_250_[#6X4:1]_epsilon0.106_rmin_half1.878.nc','cyclohexane_250_[#6X4:1]_epsilon0.111102031331_rmin_half1.88003963316.nc'] 
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.111102031331_rmin_half1.88003963316.nc','cyclohexane_250_[#6X4:1]_epsilon0.123802859878_rmin_half1.83009792013.nc','cyclohexane_250_[#6X4:1]_epsilon0.122806466703_rmin_half1.83153401667.nc','cyclohexane_250_[#6X4:1]_epsilon0.123056669259_rmin_half1.82907156612.nc'] #lowhigh
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.15_rmin_half2.5.nc','cyclohexane_250_[#6X4:1]_epsilon0.032838514744_rmin_half2.47399343003.nc','cyclohexane_250_[#6X4:1]_epsilon0.0291396893816_rmin_half2.44801411952.nc','cyclohexane_250_[#6X4:1]_epsilon0.112177820691_rmin_half2.26019955072.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.112177820691_rmin_half2.26019955072.nc','cyclohexane_250_[#6X4:1]_epsilon0.0989536390631_rmin_half2.03461350156.nc','cyclohexane_250_[#6X4:1]_epsilon0.0879851616583_rmin_half1.86229747403.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.0846735362592_rmin_half1.92520944034.nc','cyclohexane_250_[#6X4:1]_epsilon0.0998180517699_rmin_half1.8870420476.nc','cyclohexane_250_[#6X4:1]_epsilon0.124483041163_rmin_half1.90601304086.nc','cyclohexane_250_[#6X4:1]_epsilon0.11721479634_rmin_half1.85079233717.nc'] #highhigh 
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.124483041163_rmin_half1.90601304086.nc','cyclohexane_250_[#6X4:1]_epsilon0.11721479634_rmin_half1.85079233717.nc','cyclohexane_250_[#6X4:1]_epsilon0.120383091103_rmin_half1.83324791999.nc','cyclohexane_250_[#6X4:1]_epsilon0.111720699353_rmin_half1.84394011198.nc'] 
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.120383091103_rmin_half1.83324791999.nc','cyclohexane_250_[#6X4:1]_epsilon0.111720699353_rmin_half1.84394011198.nc','cyclohexane_250_[#6X4:1]_epsilon0.113291079982_rmin_half1.84034392831.nc','cyclohexane_250_[#6X4:1]_epsilon0.123476918293_rmin_half1.81040455874.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.120383091103_rmin_half1.83324791999.nc','cyclohexane_250_[#6X4:1]_epsilon0.123476918293_rmin_half1.81040455874.nc','cyclohexane_250_[#6X4:1]_epsilon0.120806343095_rmin_half1.82651886871.nc','cyclohexane_250_[#6X4:1]_epsilon0.12689920064_rmin_half1.79748919739.nc','cyclohexane_250_[#6X4:1]_epsilon0.12717137523_rmin_half1.79711005539.nc']
files = ['cyclohexane_250_[#6X4:1]_epsilon0.12689920064_rmin_half1.79748919739.nc','cyclohexane_250_[#6X4:1]_epsilon0.12717137523_rmin_half1.79711005539.nc']#highhigh
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.1102_rmin_half1.747.nc','cyclohexane_250_[#6X4:1]_epsilon0.121_rmin_half1.845.nc','cyclohexane_250_[#6X4:1]_epsilon0.123169898921_rmin_half1.80998708165.nc','cyclohexane_250_[#6X4:1]_epsilon0.122736702268_rmin_half1.81324689412.nc','cyclohexane_250_[#6X4:1]_epsilon0.122996070934_rmin_half1.8097651118.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.118_rmin_half1.666.nc','cyclohexane_250_[#6X4:1]_epsilon0.1102_rmin_half1.747.nc','cyclohexane_250_[#6X4:1]_epsilon0.0634082340145_rmin_half1.87902491641.nc']#'cyclohexane_250_[#6X4:1]_epsilon0.121_rmin_half1.845.nc']#,'cyclohexane_250_[#6X4:1]_epsilon0.149885671264_rmin_half1.84779306431.nc']#['cyclohexane_250_[#6X4:1]_epsilon0.05_rmin_half1.5.nc','cyclohexane_250_[#6X4:1]_epsilon0.118_rmin_half1.666.nc']
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080.nc','cyclohexane_250_[#6X4:1]_epsilon0.0972166482044_rmin_half1.92015303945.nc','cyclohexane_250_[#6X4:1]_epsilon0.129280592666_rmin_half1.80945011123.nc','cyclohexane_250_[#6X4:1]_epsilon0.129946677226_rmin_half1.80632698086.nc','cyclohexane_250_[#6X4:1]_epsilon0.129422275289_rmin_half1.80803528024.nc','cyclohexane_250_[#6X4:1]_epsilon0.129684432789_rmin_half1.80796234322.nc'] 
#files = ['cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080.nc','cyclohexane_250_[#6X4:1]_epsilon0.102200007296_rmin_half1.91839456869.nc','cyclohexane_250_[#6X4:1]_epsilon0.114144607104_rmin_half1.88700000237.nc','cyclohexane_250_[#6X4:1]_epsilon0.114544784073_rmin_half1.88700000176.nc','cyclohexane_250_[#6X4:1]_epsilon0.114745093562_rmin_half1.88700000404.nc','cyclohexane_250_[#6X4:1]_epsilon0.114493562595_rmin_half1.88700000523.nc']#['cyclohexane_250_[#6X4:1]_epsilon0.1094_rmin_half1.9080.nc','cyclohexane_250_[#6X4:1]_epsilon0.10220000002_rmin_half1.92580500001.nc']

# Extracting the '[#6X4:1]_epsilon<value>_rmin_half<value>' part of the strings from `files`
file_strings = [i.rsplit('.',1)[0].split('_',2)[2] for i in files]

# Creating the correct filenames for the data I want to use
# Configuration and state data trajectories for the liquid and ideal gasw simulations
file_tups_traj = [['traj_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_250_'+i+'_wNoConstraints_1fsts.nc'] for i in file_strings]
file_tups_traj_vac = [['traj_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_'+i+'_wNoConstraints_vacuum_0.8fsts.nc'] for i in file_strings]

file_tups_sd = [['StateData_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_250_'+i+'_wNoConstraints_1fsts.csv'] for i in file_strings]
file_tups_sd_vac = [['StateData_cychex_neat/Lang_2_baro10step_pme1e-5/cyclohexane_'+i+'_wNoConstraints_vacuum_0.8fsts.csv'] for i in file_strings]

# Create a list of lists for the parameters in the reference data
params = [i.rsplit('.',1)[0].rsplit('_') for i in files]
params = [(i[3][7:],i[5][4:]) for i in params]


# Set up containers for all the data I'm extracting from the trajectories
xyz_orig = [[] for i in file_tups_traj] # coordinates from liquid sims
xyz_orig_vac = [[] for i in file_tups_traj] # coordinates from gas sims
vol_orig = [[] for i in file_tups_traj] # molar volumes from liquid sims
ener_orig = [[] for i in file_tups_sd] # potential energies from liquid sims
ener_orig_vac = [[] for i in file_tups_sd] # potential energies from gas sims
vecs_orig = [[] for i in file_tups_sd] # box vectors from liquid sims
vol_box_orig = [[] for i in file_tups_sd] # box volumes from liquid sims
steps_orig_vac = [[] for i in file_tups_sd] # step number from gas sims (debugging those sims, so looking at traces)
temp_orig = [[] for i in file_tups_sd] # temperatures from liquid sims
temp_orig_vac = [[] for i in file_tups_sd] # temperatures from gas sims

# Define number of burnin samples for the liquid and gas sims
burnin = 1000#3997#1949
burnin_vac = 1000#7997#3949

print('burnin bulk = %s' %(burnin))
print('burnin vac = %s' %(burnin_vac))

print( 'Extracting data from cyclohexane neat liquid configuration trajectories')
for j,i in enumerate(file_tups_traj):
    for ii in i:            
        try:
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
param_types = ['epsilon','rmin_half']


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
steps_orig_vac_sub = [[] for i in steps_orig_vac]
temp_orig_sub = [[] for i in temp_orig]
temp_orig_vac_sub = [[] for i in temp_orig_sub]

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

    ener_orig_sub[ii] = ener_sub
    vol_orig_sub[ii] = vol_sub
    vol_box_orig_sub[ii] = vol_box_sub
    xyz_orig_sub[ii] = xyz_sub
    vecs_orig_sub[ii] = vecs_sub
    temp_orig_sub[ii] = temp_sub

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
"""
plt.figure()
plt.plot(steps_orig_vac_sub[0],ener_orig_vac_sub[0])
plt.xlabel('Timestep (units of 0.8 fs)')
plt.ylabel('Potential Energy (kJ/mole)')
#plt.tight_layout()
plt.savefig('Vacuum_potential_trace_HConstraints_10ns.png',dpi=300)
"""

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
eps_vals = np.linspace(float(argv[1]),float(argv[2]),3)
rmin_vals = np.linspace(float(argv[3]),float(argv[4]),3)

# Turn array of floats into list of strings (the forcefield editing machinery takes string inputs) 
eps_vals = [str(a) for a in eps_vals]
rmin_vals = [str(a) for a in rmin_vals]
new_states = list(product(eps_vals,rmin_vals))# List of all (eps, rmin_half) combinations 

new_states = list(set(new_states)) # uniquefy list

orig_state = params # the list of tuples from the reference parameter states

# Get rid of a state in 'new_states' if it's already in the 'orig_state' object
for i in new_states:
    if i in orig_state:
        new_states.remove(i)

N_eff_list = []
param_type_list = []
param_val_list = []

state_coords = orig_state
for i in new_states:
     state_coords.append(i)

# load up the packmol box from the liquid simulation for energy re-evaluation
filename = 'packmol_boxes/cyclohexane_250.pdb'
pdb = PDBFile(filename)

# Make containers for u_kn and E_kn matrices
u_kns = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_sub])],np.float64)
E_kns = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_sub])],np.float64)
u_kns_vac = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_vac_sub])],np.float64)
E_kns_vac = np.zeros([len(state_coords),sum([len(i) for i in ener_orig_vac_sub])],np.float64)

# Make new N_k (samples per state) objects based on subsampled energy lengths, but also 
# filling in zeros for the states we're predicting (and don't have samples for)
N_k = np.zeros(len(state_coords),np.int64)
N_k_vac = np.zeros(len(state_coords),np.int64)

for k,i in enumerate(ener_orig_sub): 
    N_k[k] = len(i)
for k,i in enumerate(ener_orig_vac_sub):
    N_k_vac[k] = len(i)

# Begin energy re-evaluations
index = 0
index_vac = 0
for ii,value in enumerate(xyz_orig_sub):
    MBAR_moves = state_coords
    print( "Number of MBAR calculations for liquid cyclohexane: %s" %(len(MBAR_moves)))
    print( "starting MBAR calculations")
    D = OrderedDict()
    for i,val in enumerate(MBAR_moves):
        D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    D_mol = {'cyclohexane' : D} 
        
    # Produce the u_kn matrix for MBAR based on the subsampled configurations
    # The `NPT=True` flag in the function automatically spits out enthalpies instead of internal energies
    E_kn, u_kn = new_param_energy(xyz_orig_sub[ii],D_mol, pdb.topology,vecs_orig_sub[ii],P=1.01,T = np.mean(temp_sub[ii]),NPT=True,V=vol_sub[ii])

    # Filling in re-evaluated energies from the iith reference configuration into the E_kn and u_kn matrices 
    curr_k = 0
    for E_n,u_n in zip(E_kn,u_kn): 
        E_kns[curr_k,index:index+len(E_n)] = E_n
        u_kns[curr_k,index:index+len(u_n)] = u_n
        curr_k += 1

    # this index will indicate where to start adding energies to the E_kn and u_kn matrices on the next iteration of the loop
    index += len(E_kn[0])

    # Same structure for re-evaluating the gas energies
    print( "Number of MBAR calculations for cyclohexane in vacuum: %s" %(len(MBAR_moves)))
    print( "starting MBAR calculations")
    D = OrderedDict()
    for i,val in enumerate(MBAR_moves):
        D['State' + ' ' + str(i)] = [["[#6X4:1]",param_types[j],val[j]] for j in range(len(param_types))]#len(state_orig))]
    D_mol = {'cyclohexane' : D}
 
    #Produce the u_kn matrix for MBAR based on the subsampled configurations
    E_kn_vac, u_kn_vac = new_param_energy_vac(xyz_orig_vac_sub[ii], D_mol, T = np.mean(temp_vac_sub[ii]))

    curr_k_vac = 0

    for E_n_vac,u_n_vac in zip(E_kn_vac,u_kn_vac):
        E_kns_vac[curr_k_vac,index_vac:index_vac+len(E_n_vac)] = E_n_vac
        u_kns_vac[curr_k_vac,index_vac:index_vac+len(u_n_vac)] = u_n_vac
        curr_k_vac += 1

    index_vac += len(E_kn_vac[0])

# number of states and total number of samples
K,N = np.shape(u_kns)
K_vac,N_vac = np.shape(u_kns_vac)


# We're going to get bootstrapped estimates of variance in order to test the stability of MBAR's asymptotic
# variance estimator. Thus, we'll define the number of bootstrap samples and make some containers for the observables
# we're bootstrapping. 
nBoots_work = 2
nBoots_work_vac = 2

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
 
#plt.figure()
#plt.hist(E_vac_boots_vt[1:,0],bins=200)
#plt.xlabel('Potential Energy (kJ/mole)')
#plt.ylabel('Frequency')
#plt.savefig('vacuum_potential_HConstraints_bootstrap_dist_10ns.png',dpi=300)
    
#plt.figure()
#plt.hist(E_boots_vt[1:,0],bins=200)
#plt.xlabel('Potential Energy (kJ/mole)')
#plt.ylabel('Frequency')
#plt.savefig('bulk_potential_bootstrap_dist.png',dpi=300)

#plt.figure()
#plt.hist(V_boots_vt[1:,0],bins=200)
#plt.xlabel('Molar volume (mL/mole)')
#plt.ylabel('Frequency')
#plt.savefig('molar_volume_bootstrap_dist.png',dpi=300)

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
    
df.to_csv('StateData_cychex_neat/Lang_2_baro10step_pme1e-5/MBAR_estimates/MBAR_estimates_[6X4:1]_eps'+argv[1]+'-'+argv[2]+'_rmin'+argv[3]+'-'+argv[4]+'_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_stabilitytest17thand18threfsof18_epshighrminhigh_Tempseries_6-5.csv',sep=';')
#df.to_csv('StateData_cychex_neat/Lang_2_baro10step_pme1e-5/MBAR_estimates/MBAR_estimates_[6X4:1]_eps'+argv[1]+'-'+argv[2]+'_rmin'+argv[3]+'-'+argv[4]+'_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_stabilitytestlast4refsof11_epslowrminhigh_Tempseries_5-27.csv',sep=';')   
#df.to_csv('StateData_cychex_neat/Lang_2_baro10step_pme1e-5/MBAR_estimates/MBAR_estimates_[6X4:1]_eps'+argv[1]+'-'+argv[2]+'_rmin'+argv[3]+'-'+argv[4]+'_baro10step_molvolPV_wNoConstraintssim_unconstrainedMBAR_MBAR1e-5PMEco_Lang_1fs_0.5nsburn_stabilitytest1ref_epslowrminhigh_Tempseries_5-12.csv',sep=';')
#with open('param_states_1fs.pkl', 'wb') as f:
#    pickle.dump(MBAR_moves, f)
#with open('u_kn_bulk_1fs.pkl', 'wb') as f:
#    pickle.dump(u_kn, f)
#with open('u_kn_vac_1fs.pkl', 'wb') as f:
#    pickle.dump(u_kn_vac, f)

