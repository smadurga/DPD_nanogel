import espressomd
import espressomd.analyze
import espressomd.electrostatics
import espressomd.accumulators
import espressomd.observables
import espressomd.polymer
import espressomd.io.writer.vtf
import espressomd.virtual_sites
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

espressomd.assert_features(['LENNARD_JONES',"DPD", "HAT"])

def Create_nanogel():
    """
    Creates the nanogel, from the files: "id_type_pos.dat" and "id_bond_ids.dat"
    
    File "id_type_pos.dat", this file contains the id, type and position of the particles.
    File "id_bond_ids.dat", this file contains the infromation of the particle bonds.

    """
    
    ########## Read id, coordinates and type information from data of a previous nanogel simulation using langevin dynamics

    datafile_path = "id_type_pos.dat"
    f = open(datafile_path,"r")
    lines=f.readlines()
    id_list = []
    type_list = []
    x_positions_list = []
    y_positions_list = []
    z_positions_list = []

    for x in lines:
        r = (x.split()[0])
        r = float(r)
        id_list.append(r)

        r = (x.split()[1])
        r = float(r)
        type_list.append(r)

        r = (x.split()[2])
        r = float(r)*(6.5/sigma)-37
        x_positions_list.append(r)

        r = (x.split()[3])
        r = float(r)*(6.5/sigma)-37
        y_positions_list.append(r)

        r = (x.split()[4])
        r = float(r)*(6.5/sigma)-37
        z_positions_list.append(r)

    id_type_positions_array = np.column_stack((id_list,type_list,x_positions_list,y_positions_list,z_positions_list))

    ########## Read id, and bond ids information from data of a previous nanogel simulation using langevin dynamics

    datafile_path = "id_bond_ids.dat"
    f = open(datafile_path,"r")
    lines=f.readlines()
    bond_1_list = []
    bond_2_list = []
    bond_3_list = []
    bond_4_list = []

    for x in lines:

        r = (x.split()[1])
        r = float(r)
        r = int(r)
        bond_1_list.append(r)

        r = (x.split()[2])
        r = float(r)
        r = int(r)
        bond_2_list.append(r)

        r = (x.split()[3])
        r = float(r)
        r = int(r)
        bond_3_list.append(r)

        r = (x.split()[4])
        r = float(r)
        r = int(r)
        bond_4_list.append(r)

    bond_ids_array = np.column_stack((id_list,type_list,bond_1_list,bond_2_list,bond_3_list,bond_4_list))
    bond_ids_array = bond_ids_array.astype(int)
    ########## Add polymer

    for i in range(len(id_type_positions_array)):
        system.part.add(type=int(id_type_positions_array[i,1]),pos=id_type_positions_array[i,2:5],mass=mass_monomer/mass_water_bead)

    ########## Add bonds between monomers

    for i in range(len(bond_ids_array)):
        if (sum(bond_ids_array[i,2:6])>0):
            if (bond_ids_array[i,1] == 0):
                system.part.by_id(i).add_bond((harmonic_bond, bond_ids_array[i,2]))
            else:
                for j in range(4): 
                    if (bond_ids_array[i,2+j] != 0):
                        system.part.by_id(i).add_bond((harmonic_bond_2, bond_ids_array[i,2+j]))

########## Modified ESPResSo functions for saving the bead positions in folded coordinates oposed to the default unfolded coordinates.

def vtf_pid_map(system, types='all'):
    """
    Generates a VTF particle index map to ESPResSo ``id``.
    This fills the gap for particle ID's as required by VMD.

    Parameters
    ----------
    system: :obj:`espressomd.system.System`
    types : :obj:`str`
        Specifies the particle types. The id mapping depends on which
        particles are going to be printed. This should be the same as
        the one used in :func:`writevsf()` and :func:`writevcf()`.
    Returns
    -------
    dict:
        A dictionary where the values are the VTF indices and the keys
        are the ESPresSo particle ``id``.
    """

    if not hasattr(types, '__iter__'):
        types = [types]
    if types == "all":
        types = [types]
    id_to_write = []
    for p in system.part:
        for t in types:
            if t in (p.type, "all"):
                id_to_write.append(p.id)
    return dict(zip(id_to_write, range(len(id_to_write))))

def writevcf2(system, fp, types='all'):
    """
    writes a VCF (VTF Coordinate Format) to a file.
    This can be used to write a timestep to a VTF file.

    Parameters
    ----------
    system: :obj:`espressomd.system.System`
    types : :obj:`str`
        Specifies the particle types. The string 'all' will write all particles
    fp : file
        File pointer to write to.

    """
    vtf_index = vtf_pid_map(system, types)
    fp.write("\ntimestep indexed\n")
    for pid, vtf_id, in vtf_index.items():
        fp.write(f"{vtf_id} {' '.join(map(str, system.part.by_id(pid).pos_folded))}\n")
        
########## Setup constants

TIME_STEP = 0.0001      #Time step used in energy minimization
LOOPS = 5000000         #Number of integrations
STEPS = 1               #Number of integrations steps between calculating the nanogel Mass centre (NMC)
id_count = 0
Bond_constant = 0.4     #N/m
Num_avogadro = 6.02214*10.**(23.)       #1/mol
Kb = 1.38065*10.**(-23.)        #J/K        
mass_monomer = 113.16       #UMA
mass_cte = 1.66054*10.**(-27.)      #Atomic mass unit in Kg
Temp = 330.     #K
epsilon = (Kb*Temp*Num_avogadro/1000)       #KJ/mol
water_density = 984.79      #Kg/m^3
Number_of_waters_in_water_bead = 5
mass_water_bead = Number_of_waters_in_water_bead*18.01528       #UMA
isothermal_compressibility = 4.4298*10**(-10)       #1/Pa
reduced_density = 3
sigma = (((reduced_density*mass_water_bead*mass_cte)/(water_density))**(1/3))*(10**10)    #Cutoff distance in Ang

#Compute density of NIPAM monomer

number_density_of_single_water_molecule = (Num_avogadro*water_density)/(mass_water_bead/(1000*Number_of_waters_in_water_bead))
number_density_of_coarsegrained_water = (Num_avogadro*water_density)/(mass_water_bead/(1000))
Water_molecule_volume = 18.01528/(water_density*Num_avogadro*1000)*(10**30)  #Ang^3
NIPAM_monomer_volume = ((4*np.pi*(3.25**3))/3) #Ang^3
NIPAM_density_reduced =  ((5*Water_molecule_volume*3)/NIPAM_monomer_volume)

#Reduced units constants

NIPAM_density_reduced =  ((5*Water_molecule_volume*3)/NIPAM_monomer_volume)
BOX_L = 22.5
dimensionless_inverse_compressibility = 1/(number_density_of_single_water_molecule*Kb*Temp*isothermal_compressibility)
T_reduced = 1. 
Reduced_Bond_constant = (Bond_constant*sigma*sigma*Num_avogadro)/(epsilon*1000.0*10.**(20.))
KT_reduced = T_reduced
time_for_step = np.sqrt((mass_water_bead*mass_cte*Num_avogadro)/(1000.*epsilon))*sigma*10.**(-10.)
Gamma_reduced = 9/(2*KT_reduced)
X_ij = 0.5 + 35.2*(1-(308.3/Temp))

#Compute Pressure in reduced units, a_ww, a_mm and a_wm

Pressure = (a_ww*0.101*(3**2) + 3)
a_ww = Number_of_waters_in_water_bead*((dimensionless_inverse_compressibility-1))/(reduced_density*2*0.101)
a_mm = (Pressure-NIPAM_density_reduced)/(0.101*(NIPAM_density_reduced**2))
a_wm = np.sqrt(a_mm*a_ww) + (X_ij*(Pressure/(0.0454*(a_ww*3 + a_mm*NIPAM_density_reduced))))

#Compute the number of water beads in the system

Total_water_beads = int(reduced_density*(BOX_L**3)-439*(mass_monomer/mass_water_bead))

########## Open trajectory file to save the trajectory

fp = open('trajectory.vtf', mode='w+t')

########## System setup

system = espressomd.System(box_l=3*[BOX_L], periodicity=[True, True, True])

########### Lennard-Jones, WCA interactions

system.non_bonded_inter[0, 0].hat.set_params(F_max=a_mm, cutoff=1.0)
system.non_bonded_inter[0, 1].hat.set_params(F_max=a_mm, cutoff=1.0)
system.non_bonded_inter[1, 1].hat.set_params(F_max=a_mm, cutoff=1.0)
system.non_bonded_inter[1, 5].hat.set_params(F_max=a_wm, cutoff=1.0)
system.non_bonded_inter[0, 5].hat.set_params(F_max=a_wm, cutoff=1.0)
system.non_bonded_inter[5, 5].hat.set_params(F_max=a_ww, cutoff=1.0)

system.non_bonded_inter[0, 0].dpd.set_params(weight_function=0, gamma=Gamma_reduced, r_cut=1.0,
    trans_weight_function=0, trans_gamma=Gamma_reduced, trans_r_cut=1.0)
system.non_bonded_inter[1, 1].dpd.set_params(weight_function=0, gamma=Gamma_reduced, r_cut=1.0,
    trans_weight_function=0, trans_gamma=Gamma_reduced, trans_r_cut=1.0)
system.non_bonded_inter[1, 0].dpd.set_params(weight_function=0, gamma=Gamma_reduced, r_cut=1.0,
    trans_weight_function=0, trans_gamma=Gamma_reduced, trans_r_cut=1.0)
system.non_bonded_inter[0, 5].dpd.set_params(weight_function=0, gamma=Gamma_reduced, r_cut=1.0,
    trans_weight_function=0, trans_gamma=Gamma_reduced, trans_r_cut=1.0)
system.non_bonded_inter[1, 5].dpd.set_params(weight_function=0, gamma=Gamma_reduced, r_cut=1.0,
    trans_weight_function=0, trans_gamma=Gamma_reduced, trans_r_cut=1.0)
system.non_bonded_inter[5, 5].dpd.set_params(weight_function=0, gamma=Gamma_reduced, r_cut=1.0,
    trans_weight_function=0, trans_gamma=Gamma_reduced, trans_r_cut=1.0)

########## Bonded interactions 

harmonic_bond = espressomd.interactions.HarmonicBond(k=Reduced_Bond_constant, r_0=(6.5/sigma))
system.bonded_inter.add(harmonic_bond)
harmonic_bond_2 = espressomd.interactions.HarmonicBond(k=Reduced_Bond_constant, r_0=(6.5/sigma))
system.bonded_inter.add(harmonic_bond_2)

Create_nanogel()
        
########## Add water beads

system.part.add(type=[5]*(Total_water_beads),pos=(([1.,1.,1.]-np.random.random(((Total_water_beads), 3))))*(BOX_L))

########## write structure block as header

espressomd.io.writer.vtf.writevsf(system, fp, types="all")

########## write initial positions as coordinate block

writevcf2(system, fp,types="all")

########## Energy minimization

system.time_step = TIME_STEP
system.integrator.set_steepest_descent(
    f_max=1.0*10**(-8),
    gamma=10,
    max_displacement=0.1)
for i in range(20):
    system.integrator.run(60)
writevcf2(system, fp,types="all")

#Set skin for verlet list

system.integrator.set_vv()
system.thermostat.set_dpd(kT = KT_reduced, seed = 42)

skin = system.cell_system.tune_skin(min_skin=0.0,max_skin=BOX_L/10,tol=10.**(-10),int_steps=64)
system.cell_system.skin = skin

TIME_STEP = 0.01        #Time step used in thermalization and production run
system.time_step = TIME_STEP

########## Theramlization
    
for i in range(100):
    system.integrator.run(10000)

########## Set rdf accumulators as well as the temperature, energy, time and gyration radi array

s_0  = list(system.part.select(type=0).id)
s_1 = list(system.part.select(type=1).id)
s_01 = s_0+s_1

rdf_obs_ww = espressomd.observables.RDF(ids1=system.part.select(type=5).id, min_r=0, max_r=system.box_l[0]/2.0, n_r_bins=int((5)/0.01))
rdf_acc_ww = espressomd.accumulators.MeanVarianceCalculator(obs=rdf_obs_ww, delta_N=1000)

rdf_obs_wm = espressomd.observables.RDF(ids1=system.part.select(type=5).id,ids2 = s_01, min_r=0, max_r=system.box_l[0]/2.0, n_r_bins=int((5)/0.01))
rdf_acc_wm = espressomd.accumulators.MeanVarianceCalculator(obs=rdf_obs_wm, delta_N=1000)

rdf_obs_mm = espressomd.observables.RDF(ids1=s_01, min_r=0, max_r=system.box_l[0]/2.0, n_r_bins=int((5)/0.01))
rdf_acc_mm = espressomd.accumulators.MeanVarianceCalculator(obs=rdf_obs_mm, delta_N=1000)

NMC = (404*np.array(system.analysis.center_of_mass(p_type=0)) + 35*np.array(system.analysis.center_of_mass(p_type=1)))/(439)        #Computing the inital nanogel Mass centre
Mass_center_particle = system.part.add(type=[6]*(1),pos=[NMC])      #Set a non interactive particle in the nanogel Mass centre
Mass_center_particle.fix = [True,True,True]     #Fix the nanogel Mass centre particle to ensure no displacement (just in case, since it is not included in any interaction)

rdf_obs_NMC_ww = espressomd.observables.RDF(ids1=[34059],ids2=system.part.select(type=5).id, min_r=0, max_r=system.box_l[0]/2.0, n_r_bins=int((5)/0.01))
rdf_acc_NMC_ww = espressomd.accumulators.MeanVarianceCalculator(obs=rdf_obs_NMC_ww, delta_N=1000)

system.auto_update_accumulators.add(rdf_acc_ww)
system.auto_update_accumulators.add(rdf_acc_wm)
system.auto_update_accumulators.add(rdf_acc_mm)
system.auto_update_accumulators.add(rdf_acc_NMC_ww)

temperature = np.zeros(int(LOOPS))
energies = np.zeros(int(LOOPS))
time = np.zeros(int(LOOPS))
rgs_results = np.zeros(int(LOOPS))

########## Production Run

x = 0

for i in range(LOOPS):
    system.integrator.run(STEPS,reuse_forces=True)
    NMC = (404*np.array(system.analysis.center_of_mass(p_type=0)) + 35*np.array(system.analysis.center_of_mass(p_type=1)))/(439)        #Compute the new position of the nanogel Mass centre
    Mass_center_particle.update({'pos': NMC})       #Update the NMC position
    energies[i]= (system.analysis.energy()['total'])
    temperature[i]= ((system.analysis.energy()['kinetic'])*(2./((3.*(Total_water_beads+439)))))
    rgs_results[i] = system.analysis.calc_rg(chain_start=0,number_of_chains=1,chain_length=439)[0]
    time[i] = i*TIME_STEP*time_for_step*10.**(12.)
    x = x + 1
    if x == 50000:
        writevcf2(system, fp,types="all")
        x = 0

#Obtain rdf results

rdf_ww = rdf_acc_ww.mean()
rs_ww = rdf_obs_ww.bin_centers()

rdf_wm = rdf_acc_wm.mean()
rs_wm = rdf_obs_wm.bin_centers()

rdf_mm = rdf_acc_mm.mean()
rs_mm = rdf_obs_mm.bin_centers()

rdf_NMC_ww  = rdf_acc_NMC_ww.mean()
rs_NMC_ww = rdf_obs_NMC_ww.bin_centers()

#Turn off thermostat and accumulators

system.auto_update_accumulators.clear()
system.thermostat.turn_off()

########## Untransfrom reduced units into real units and save results in .dat files

ts = np.arange(0, int(LOOPS))

datafile_path = "Results.dat"
np.savetxt(datafile_path , np.c_[time,rgs_results*sigma,energies*epsilon,temperature*(1000.0*epsilon)/(Num_avogadro*Kb)],fmt=["%7f","%7f","%7f","%7f"])
datafile_path = "GDR_wm_Results.dat"
np.savetxt(datafile_path , np.c_[rs_wm*sigma,rdf_wm],fmt=["%7f","%7f"])
datafile_path = "GDR_ww_Results.dat"
np.savetxt(datafile_path , np.c_[rs_ww*sigma,rdf_ww],fmt=["%7f","%7f"])
datafile_path = "GDR_mm_Results.dat"
np.savetxt(datafile_path , np.c_[rs_mm*sigma,rdf_mm],fmt=["%7f","%7f"])
datafile_path = "GDR_NMC_ww_Results.dat"
np.savetxt(datafile_path , np.c_[rs_NMC_ww*sigma,rdf_NMC_ww],fmt=["%7f","%7f"])


