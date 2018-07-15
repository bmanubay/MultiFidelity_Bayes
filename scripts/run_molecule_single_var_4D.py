#!/bin/env python

import time
import simtk.openmm as mm
from simtk.openmm import app
from simtk.openmm import Platform
from simtk.unit import *
import numpy as np
from mdtraj.reporters import NetCDFReporter
# Import the SMIRNOFF forcefield engine and some useful tools
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import get_data_filename, extractPositionsFromOEMol, generateTopologyFromOEMol
import openmmtools.integrators as ommtoolsints
from openeye import oechem
import sys
import numpy as np

molname = [sys.argv[1]]
mol_filename = ['monomers/'+m+'.mol2' for m in molname]
time_step = 0.8 #Femtoseconds
temperature = 293.15 #kelvin
friction = 1 # per picosecond
num_steps = 4000000
trj_freq = 1000 #steps
data_freq = 1000 #steps

# Load OEMol
for ind,j in enumerate(mol_filename):
    mol = oechem.OEGraphMol()
    ifs = oechem.oemolistream(j)
    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
    oechem.OEReadMolecule(ifs, mol )
    oechem.OETriposAtomNames(mol)

    # Get positions
    coordinates = mol.GetCoords()
    natoms = len(coordinates)
    positions = np.zeros([natoms,3], np.float64)
    for index in range(natoms):
        (x,y,z) = coordinates[index]
        positions[index,0] = x
        positions[index,1] = y
        positions[index,2] = z
    positions = Quantity(positions, angstroms)
    
    
    # Load forcefield
    forcefield = ForceField(get_data_filename('forcefield/smirnoff99Frosst.ffxml'))

    # Define system
    topology = generateTopologyFromOEMol(mol)
    params = forcefield.getParameter(smirks='[#1:1]-[#8]')
    params['rmin_half']='0.01'
    params['epsilon']='0.01'
    forcefield.setParameter(params, smirks='[#1:1]-[#8]')
    system = forcefield.createSystem(topology, [mol])
    
    smirkseries1 = sys.argv[2]
    eps = sys.argv[3]
    rmin = sys.argv[4]
    epsval1 = sys.argv[5]
    rminval1 = sys.argv[6]

    smirkseries2 = sys.argv[7]
    epsval2 = sys.argv[8]
    rminval2 = sys.argv[9]

    param1 = forcefield.getParameter(smirks=smirkseries1)
    param1[eps] = epsval1
    param1[rmin] = rminval1
    forcefield.setParameter(param1, smirks=smirkseries1)

    param2 = forcefield.getParameter(smirks=smirkseries2)
    param2[eps] = epsval2
    param2[rmin] = rminval2
    forcefield.setParameter(param2, smirks=smirkseries2)

    #Do simulation
    integrator = ommtoolsints.LangevinIntegrator(temperature*kelvin, friction/picoseconds, time_step*femtoseconds)
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision': 'mixed','DeterministicForces': 'true'}
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature*kelvin)
    netcdf_reporter = NetCDFReporter('traj_cychex_neat/Lang_2_baro10step_pme1e-5/'+molname[ind]+'_'+smirkseries1+'_'+eps+epsval1+'_'+rmin+rminval1+'_'+smirkseries2+'_'+eps+epsval2+'_'+rmin+rminval2+'_wNoConstraints_vacuum_0.8fsts.nc', trj_freq)
    simulation.reporters.append(netcdf_reporter)
    simulation.reporters.append(app.StateDataReporter(sys.stdout, data_freq, step=True, potentialEnergy=True, temperature=True))
    simulation.reporters.append(app.StateDataReporter('StateData_cychex_neat/Lang_2_baro10step_pme1e-5/'+molname[ind]+'_'+smirkseries1+'_'+eps+epsval1+'_'+rmin+rminval1+'_'+smirkseries2+'_'+eps+epsval2+'_'+rmin+rminval2+'_wNoConstraints_vacuum_0.8fsts.csv', data_freq, step=True, potentialEnergy=True, temperature=True))
    print("Starting simulation")
    start = time.clock()
    simulation.step(num_steps)
    end = time.clock()

    print("Elapsed time %.2f seconds" % (end-start))
    netcdf_reporter.close()
    print("Done!")

