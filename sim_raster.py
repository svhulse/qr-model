import numpy as np
import pickle as pkl
import tqdm
import multiprocessing as mp

from qrmodel import Model, SimRaster

'''
Model for single pathogen with only specific resistance with frequency
dependent transmission
'''
M0 = SimRaster()
M0.name = 'one_path_no_q'
M0.model = Model(transmission = 'fd',
		mu=0.2, gamma=0.01, beta_e=1,
		q_init=1, R_int=0, h=0.5,
		N_iter=2)
M0.x_var = 'r'
M0.y_var = 'c'
M0.x_range = [0, 1]
M0.y_range = [0, 0.5]

'''
Model for single pathogen with general and specific resistance with frequency
dependent transmission

Begins with no specific resistance, allow specific resistance to reach 
equilibrium and then introduce the resistant genotype

Parametes varied are cost and strength of specific resistance
'''
M1 = SimRaster()
M1.name = 'one_path'
M1.model = Model(transmission = 'fd',
		mu=0.2, gamma=0.01, beta_e=1,
		q_init=0.65, R_int=1, h=0.5,
		N_iter=60)
M1.x_var = 'r'
M1.y_var = 'c'
M1.x_range = [0, 1]
M1.y_range = [0, 0.5]

'''
Model for single pathogen with general and specific resistance with frequency
dependent transmission

Begins with no general resistance, but both the susceptible and resistant
genotypes. Then allows general resistance to evolve to equilibrium

Parametes varied are cost and strength of specific resistance

'''
M2 = SimRaster()
M2.name = 'one_path_q0'
M2.model = Model(transmission = 'fd',
		mu=0.2, gamma=0.01, beta_e=1,
		q_init=0.99, R_int=1, h=0.5,
		N_iter=60)
M2.x_var = 'r'
M2.y_var = 'c'
M2.x_range = [0, 1]
M2.y_range = [0, 0.5]

'''
Model for single pathogen with general and specific resistance with density
dependent transmission

Begins with no specific resistance, allow specific resistance to reach 
equilibrium and then introduce the resistant genotype

Parametes varied are cost and strength of specific resistance
'''	
M3 = SimRaster()
M3.name = 'one_path_dd'
M3.model = Model(transmission = 'dd', 
		mu=0.2, gamma=0.01, beta_e=0.01,
		q_init=0.65, R_int=10, h=0.5,
		N_iter=50)
M3.x_var = 'r'
M3.y_var = 'c'
M3.x_range = [0, 1]
M3.y_range = [0, 0.5]

'''
Model for two pathogens with general and specific resistance with frequency
dependent transmission

Begins with no general resistance, but both the susceptible and resistant
genotypes as well as both pathogens. Then allows general resistance to evolve 
to equilibrium

Parametes varied are cost and strength of specific resistance
'''
M4 = SimRaster()
M4.name = 'two_path_q0'
M4.model = Model(transmission = 'fd',
		mu=0.2, gamma=0.01, beta_e=1,
		q_init=0.99, R_int=1, Y2_int=1,
		h=0.5, beta_f=0.6, N_iter=60)
M4.x_var = 'r'
M4.y_var = 'c'
M4.x_range = [0, 1]
M4.y_range = [0, 0.5]

'''
Model for two pathogens with general and specific resistance with frequency
dependent transmission

Begins with just S with intermediate general resistance, allows evolution to
optimal q, then introduces R, then the foreign pathogen.

Parametes varied are cost and strength of specific resistance
'''

M5 = SimRaster()
M5.name = 'two_path'
M5.model = Model(transmission = 'fd', 
		mu=0.2, gamma=0.01, beta_e=1,
		q_init=0.65, R_int=1, Y2_int=50, h=0.5,
		beta_f=0.6, N_iter=140)
M5.x_var = 'r'
M5.y_var = 'c'
M5.x_range = [0, 1]
M5.y_range = [0, 0.5] 

'''
Model for two pathogens with general and specific resistance with frequency
dependent transmission

Begins with no general resistance, but both the susceptible and resistant
genotypes as well as both pathogens. Then allows general resistance to evolve 
to equilibrium

Parametes varied are cost of specific resistance and foreign pathogen 
transmission
'''
M6 = SimRaster()
M6.name = 'two_path_q0'
M6.model = Model(transmission = 'fd', 
		mu=0.2, r=0.2, gamma=0.01, beta_e=1,
		q_init=0.99, R_int=1, Y2_int=1, h=0.5,
		N_iter=140)
M6.x_var = 'beta_f'
M6.y_var = 'c'
M6.x_range = [0, 1]
M6.y_range = [0, 0.5]

'''
Model for two pathogens with general and specific resistance with density
dependent transmission

Begins with no general resistance, but both the susceptible and resistant
genotypes as well as both pathogens. Then allows general resistance to evolve 
to equilibrium

Parametes varied are cost of specific resistance and foreign pathogen 
transmission
'''
M7 = SimRaster()
M7.name = 'two_path_dd_2'
M7.model = Model(transmission = 'dd', 
		mu=0.2, r=0.2, gamma=0.01, beta_e=0.01,
		q_init=0.8, R_int=10, Y2_int=60, h=0.5,
		N_iter=160)
M7.x_var = 'beta_f'
M7.y_var = 'c'
M7.x_range = [0, 0.01]
M7.y_range = [0, 0.5]

'''
Model for two pathogens with general and specific resistance with frequency
dependent transmission

Begins with just S with intermediate general resistance, allows evolution to
optimal q, then introduces R, then the foreign pathogen.

Parametes varied are cost and strength of specific resistance
'''

M8 = SimRaster()
M8.name = 'foreign_path'
M8.model = Model(transmission = 'fd', 
		mu=0.2, gamma=0.01, r=0.2, beta_e=1,
		q_init=0.65, R_int=1, Y2_int=50, 
		h=0.5, N_iter=140)
M8.x_var = 'beta_f'
M8.y_var = 'c'
M8.x_range = [0, 1]
M8.y_range = [0, 0.5]

sim = M4
data_fname = '/home/sam/Projects/QR_Model/Data/' + sim.name + '_data.p'

n_x = 100 #Number of steps for variable 1
n_y = 100 #Number of steps for variable 2

def pass_to_sim(kwargs):
	return sim.model.run_sim(**kwargs)

if __name__ == '__main__':
	x_vals = np.linspace(sim.x_range[0], sim.x_range[1], n_x)
	y_vals = np.linspace(sim.y_range[0], sim.y_range[1], n_y)
	
	coords = []
		
	for i in range(n_x):
		for j in range(n_y):
			coords.append((i,j))
		
	#Define dictionary of parameter values to pass to multiprocessing
	params = [{sim.x_var:x_vals[run[0]], sim.y_var:y_vals[run[1]]} for run in coords]

	#Run simluations for 4 core processor
	pool = mp.Pool(processes=4)	

	results = []
	for result in tqdm.tqdm(pool.imap(pass_to_sim, params), total=len(params)):
		results.append(result)

	raster = []
	for i in range(n_x):
		inds = [j for j in range(len(coords)) if coords[j][1] == i]
		raster.append([results[j] for j in inds])

	with open(data_fname, 'wb') as f:
		pkl.dump([results, params], f)