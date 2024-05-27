import espressomd
import espressomd.analyze
import espressomd.electrostatics
import espressomd.accumulators
import espressomd.observables
import espressomd.polymer
import espressomd.io.writer.vtf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

espressomd.assert_features(['LENNARD_JONES'])

def build_chain(system, Monomers_in_chain, M_previous, translation, id_count, average_charges, qe, cut_generation_dist):
    """
    Builds linear chains of the polymer and adds the bonded interactions such as bond, angles and dihedrals. This function 
    allows the creation of chains with an average number of charges that are spread randomly through the chain but avoiding consecutive
    charged beads.
    
    Inputs: 
    
    system: This is the ESPResSo system
    Monomers_in_chain (int): Number of monomers in a chain between crosslinkers.
    M_previous (np.array [1,3]): Is the coordinates of the last monomer added to the system.
    translation (np.array [1,3]): Vector used to compute the new monomer coordinates
    average_charges (int): Number of averages charge in a builded chaing.
    cut_generation_dist (float): Radius from teh center of the nanogel at which no monomers or crosslinkers are added.
    
    Outputs:
    
    qe (int): Number of charges in the builded chain
    id_count (int): counter for the number of monomers added and their respective id in the ESPResSo system.
    
    """ 
    rng = np.random.default_rng()
        
    #Select ammount of charges in the chain
    
    num = rng.random()
    
    if (num <0.25):
        Charged_beads = 0
    elif (num >= 0.25 and num < 0.75):
        Charged_beads = 0
    elif (num >= 0.75):
        Charged_beads = 0 
        
    #List for charged beads selection
    
    Charge_bead_selected = np.arange(0,Monomers_in_chain)
    Charged_beads_list = []
    for i in range(Charged_beads):
        num = int(rng.random()*7)
        if (i == 0):
            Charged_beads_list.append(num)
        else:
            num = int(rng.random()*7)
            for j in range(len(Charged_beads_list)):
                while (num in Charged_beads_list):
                    num = int(rng.random()*7)
            Charged_beads_list.append(num) 
    
    #Check for consecutive charge beads and select one bead to remain charged if consecutive charged beads are found
    
    deleted = 0
    deletion_list = []
    for i in range(len(Charged_beads_list)):
        for j in range(len(Charged_beads_list)-i):
            k = i+j
            if (Charged_beads_list[i]+1 == Charged_beads_list[k] or Charged_beads_list[i]-1 == Charged_beads_list[k]):
                deletion_list.append(i)
                deleted = deleted + 1
    if (deleted > 0):
        for x in deletion_list:
            del Charged_beads_list[x]
        
        #Remove from numpy array charged beads and beads near a charge
        
        Delete_charge_list = Charged_beads_list.copy()
        adjacent_list = [x+1 for x in Delete_charge_list]
        adjacent_list.extend([x-1 for x in Delete_charge_list])
        Delete_charge_list.extend(adjacent_list)
        if (8 in Delete_charge_list):
            Delete_charge_list.remove(8)
        if (-1 in Delete_charge_list):
            Delete_charge_list.remove(-1)
        Charge_bead_selected = np.delete(Charge_bead_selected, Delete_charge_list, None)
        
        #Add Charges deleted from been consecutive into non consecutive and non charged beads

        for i in range(deleted):
            num = int(rng.random()*np.size((Charge_bead_selected)))  ####Asegurar que da 2 no 1 en array de 2 elementos
            Value_deleted = Charge_bead_selected[num]
            Charged_beads_list.append(Value_deleted)
            Charge_bead_selected = np.delete(Charge_bead_selected, [num], None)
            if (Value_deleted == 7):
                Charge_bead_selected = np.delete(Charge_bead_selected, np.where(Charge_bead_selected == Value_deleted-1), None)
            elif (Value_deleted == 0):
                Charge_bead_selected = np.delete(Charge_bead_selected, np.where(Charge_bead_selected == Value_deleted+1), None)
            else:
                Charge_bead_selected = np.delete(Charge_bead_selected, np.where(Charge_bead_selected == Value_deleted+1), None)
                Charge_bead_selected = np.delete(Charge_bead_selected, np.where(Charge_bead_selected == Value_deleted-1), None)

    #Add monomers
  
    for n in range(Monomers_in_chain):   
        if (np.linalg.norm(M_previous+translation) <= cut_generation_dist): 
            if (n in Charged_beads_list):  
                system.part.add(type=2,pos=M_previous+translation,q=-1)
                M_previous = system.part.by_id(id_count).pos
                qe = qe + 1   
            else:
                system.part.add(type=0,pos=M_previous+translation)
                M_previous = system.part.by_id(id_count).pos

            #Add bonded interactions
             
            if (n>0):
                system.part.by_id(id_count-1).add_bond((harmonic_bond, id_count))
            
            #Keep track of monomers id    
                
            id_count =  id_count + 1

    return id_count, qe
    
def build_nanogel(cut_generation_dist,id_count,num_q):
    """
    Functions taht builds the nanogel from the crosslinkers coordinates found in the file positions.npy.
    Uses the build_chain function to create the monomers connecting the crosslinkers.
    
    Inputs: 
    
    cut_generation_dist (float): Radius from teh center of the nanogel at which no monomers or crosslinkers are added.
    
    Outputs:
    
    id_count (int): counter for the number of monomers added and their respective id in the ESPResSo system.
    
    """ 
    
    #Open positions of the crosslinkers
    
    with open('positions.npy', 'rb') as f:
        crosslinkers_pos = np.load(f)       #Array with diamond superlattice from pyiron library

    crosslinker_union_dist = Monomers_distance*lattice_length*np.sqrt(3)/4            

    #Seek for pairs within distance of crosslinker union

    crosslinkers_id_array = np.zeros(1)

    crosslinkers_pos_array = np.zeros([1,3])
    chain_count = 0
    num_q = 0
    crosslinkers_in_polymer_builded = 0
    for i in range(crosslinkers_pos.shape[0]):
        j = i+1
        dist2 = np.linalg.norm(crosslinkers_pos[i,:])
        if (dist2 <= cut_generation_dist):
            system.part.add(type=1,pos=crosslinkers_pos[i,:])
            crosslinkers_in_polymer_builded = crosslinkers_in_polymer_builded + 1
            crosslinkers_id_array = np.concatenate((crosslinkers_id_array, [id_count]))
            crosslinkers_pos_array = np.concatenate((crosslinkers_pos_array, [crosslinkers_pos[i,:]]))
            id_count = id_count + 1
        for k in range(crosslinkers_pos.shape[0]-(i+1)):
            dist = np.linalg.norm(crosslinkers_pos[i,:]-crosslinkers_pos[j,:])
            if (dist<=crosslinker_union_dist and dist > 0.0):
                chain_count = chain_count + 1
                #Compute translation vector for creating a chain between crosslinkers
                translation = (crosslinkers_pos[j,:]-crosslinkers_pos[i,:])/(Monomers_in_chain+1)
                #Add chain between crosslinkers with bonds
                id_count, num_q = build_chain(system, Monomers_in_chain, crosslinkers_pos[i,:], translation, id_count, average_charges, num_q, cut_generation_dist)

            j=j+1
    crosslinkers_id_array = np.delete(crosslinkers_id_array,0,0)
    crosslinkers_pos_array = np.delete(crosslinkers_pos_array,0,0)

    # Add bonds between crosslinkers and chains

    p = system.part.all()
    All_part_array = p.pos
    for i in range(crosslinkers_pos_array.shape[0]):
        j = 0  
        for k in range(All_part_array.shape[0]):
            dist = np.linalg.norm(crosslinkers_pos_array[i,:]-All_part_array[j,:])
            if (dist<=Crosslinkers_distance and dist > 0.0001):
                system.part.by_id(crosslinkers_id_array[i]).add_bond((harmonic_bond_2, j))         
            j=j+1
    return id_count,num_q

def save_id_type_pos_bond_id (id_count):
    """
    Functions taht saves in two files all the nangoel relevant iformation after thermalization. 
    First file "id_type_pos.dat", this file stores the id, type and position of the particles.
    Second file "id_bond_ids.dat", this file stores the infromation of the particle bonds.
    
    Input:
    
    id_count (int): counter for the number of monomers added and their respective id in the ESPResSo system.
    
    """ 
    positions = np.zeros((id_count,3))
    id_array = np.zeros((id_count))
    type_array = np.zeros((id_count))
    bond_id_array = np.zeros((id_count,4))

    for i in range(id_count):
        positions[i,:] = system.part.select(id = i).pos
        id_array[i] =  system.part.select(id = i).id
        type_array[i] = system.part.select(id = i).type
        if (type_array[i] == 0 and len([element for tupl in system.part.select(id = i).bonds for element in tupl])>=1):
            bond,bond_id = [element for tupl in system.part.select(id = i).bonds for element in tupl][0]
            bond_id_array[i,:] = 0
            bond_id_array[i,0] = bond_id
        else:
            bond_id_array[i,:] = 0
            for j in range(len([element for tupl in system.part.select(id = i).bonds for element in tupl])):
                bond,bond_id = [element for tupl in system.part.select(id = i).bonds for element in tupl][j]
                bond_id_array[i,j] = bond_id

    datafile_path = "id_type_pos.dat"
    np.savetxt(datafile_path , np.c_[id_array,type_array,positions[:,0],positions[:,1],positions[:,2]],fmt=["%10f","%10f","%10f","%10f","%10f"])

    datafile_path = "id_bond_ids.dat"

    np.savetxt(datafile_path , np.c_[id_array,bond_id_array[:,0],bond_id_array[:,1],bond_id_array[:,2],bond_id_array[:,3]],fmt=["%10f","%10f","%10f","%10f","%10f"])

########## Setup constants

average_charges = 0     #Set to 0. The nanogel simulated is neutral
TIME_STEP = 0.0001      #Time step used in energy minimization
id_count = 0
num_q = 0
Bond_constant = 0.4     #N/m
sigma = 6.5    #Angstroms
epsilon = 2.475042    #KJ/mol
Num_avogadro = 6.02214*10.**(23.)       #1/mol
Kb = 1.38065*10.**(-23.)        #J/K 
mass = 113.16       #Mass of the monomer beads in UMA
mass_cte = 1.66054*10.**(-27.)      #Atomic mass unit in Kg

#Constants that depende on Temperature

Temp = 330.     #K
Diffusion_coeficient = 4.5114*10.**(-9.)        #(m^2)/s. Water self difusion coefficient
Gamma = (Kb*Temp)/(Diffusion_coeficient*mass*mass_cte)      #Langevin dynamics damping constant (1/s)

#Constants for the hydrophobic interaction
          
T_epsilon = 307.5       #K
k_epsilon = 0.0667      #1/K
hidrophobic_epsilon_max = 3.3121        #KJ/mol
hidrophobic_range = 0.9     #nm
hidrophobic_k = 10.99/hidrophobic_range     #1/nm
hidrophobic_epsilon = (hidrophobic_epsilon_max/2)*(1+np.tanh(k_epsilon*(Temp-T_epsilon)))       #KJ/mol

#Reduced units constants

T_reduced = (Num_avogadro*Kb*Temp)/(1000.0*epsilon) 
Reduced_Bond_constant = (Bond_constant*sigma*sigma*Num_avogadro)/(epsilon*1000.0*10.**(20.))
KT_reduced = T_reduced 
time_for_step = np.sqrt((mass*mass_cte*Num_avogadro)/(1000.*epsilon))*sigma*10.**(-10.)
Gamma_reduced = Gamma*time_for_step
hidrophobic_epsilon_reduced = hidrophobic_epsilon/epsilon
hidrophobic_range_reduced = (hidrophobic_range*10)/sigma
hidrophobic_k_reduced = (hidrophobic_k*sigma)/10

#Nanogel generation constants in reduced units

cut_generation_dist = 20.
Monomers_in_chain = 8
Monomers_distance = 1.0                                                          
lattice_length = Monomers_distance*((Monomers_in_chain+1)*4)/np.sqrt(3)     #Length between crosslinkers in the nanogel
Crosslinkers_distance = 1.0     #Length between crosslinkersand monomers

########## Add hidrophobic interactions

#Energy and force array for the hidrophobic interactions table

Hidrophobic_energy = np.fromfunction(lambda i ,j: -(hidrophobic_epsilon_reduced/2)*(1-np.tanh(hidrophobic_k_reduced*((i*(2.3/1000))-hidrophobic_range_reduced))) ,(1001,1), dtype=float)
Hidrophobic_force = np.fromfunction(lambda i ,j: -hidrophobic_k_reduced*(hidrophobic_epsilon_reduced/2)*(1/(np.cosh(hidrophobic_k_reduced*((i*(2.3/1000))-hidrophobic_range_reduced))**2)) ,(1001,1), dtype=float)

########## File open trajextory for save trajectory

fp = open('trajectory.vtf', mode='w+t')

########## System setup

BOX_L = 100
system = espressomd.System(box_l=3 * [BOX_L])
       
##########  Lennard-Jones, WCA interactions

system.non_bonded_inter[0, 0].tabulated.set_params(min= 0, max= 2.3, energy= Hidrophobic_energy , force= Hidrophobic_force)
system.non_bonded_inter[0, 1].tabulated.set_params(min= 0, max= 2.3, energy= Hidrophobic_energy , force= Hidrophobic_force)
system.non_bonded_inter[1, 1].tabulated.set_params(min= 0, max= 2.3, energy= Hidrophobic_energy , force= Hidrophobic_force)
system.non_bonded_inter[0, 0].wca.set_params(epsilon=1.0, sigma=1.0)
system.non_bonded_inter[0, 1].wca.set_params(epsilon=1.0, sigma=1.0)
system.non_bonded_inter[1, 1].wca.set_params(epsilon=1.0, sigma=1.0)

##########  Bonded interactions 

harmonic_bond = espressomd.interactions.HarmonicBond(k=Reduced_Bond_constant, r_0=(1.0))
system.bonded_inter.add(harmonic_bond)
harmonic_bond_2 = espressomd.interactions.HarmonicBond(k=Reduced_Bond_constant, r_0=(1.0))
system.bonded_inter.add(harmonic_bond_2)

########## Add polymer

id_count,num_q = build_nanogel(cut_generation_dist,id_count,num_q)

########## write structure block as header

espressomd.io.writer.vtf.writevsf(system, fp)

########## write initial positions as coordinate block

espressomd.io.writer.vtf.writevcf(system, fp)

########## Energy minimization

system.time_step = TIME_STEP
system.integrator.set_steepest_descent(
    f_max=1.0*10**(-8),
    gamma=Gamma_reduced,
    max_displacement=0.1)
for i in range(200):
    system.integrator.run(30)
espressomd.io.writer.vtf.writevcf(system, fp,types="all")

#Set skin for verlet list

system.integrator.set_vv()
system.thermostat.set_langevin(kT=KT_reduced, gamma=Gamma_reduced/10, seed=42)

skin = system.cell_system.tune_skin(min_skin=0.0,max_skin=BOX_L/100,tol=10.**(-10),int_steps=1024)
system.cell_system.skin = skin

TIME_STEP = 0.01        #Time step used in thermalization and production run
system.time_step = TIME_STEP

########## Theramlization

for i in range(100):
    system.integrator.run(10000)
    
system.thermostat.turn_off()

########## write finañ positions as coordinate block

espressomd.io.writer.vtf.writevcf(system, fp)

########## Saves id, type, pos and bond_id on arrays

save_id_type_pos_bond_id(id_count)


