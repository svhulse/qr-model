import numpy as np

__author__ = 'Samuel Hulse'

class Model:
	'''
	The Model class is used to define a simulation for the QR host-pathogen
	model. It allows for both density-dependent and frequency-dependent disease
	transmission. Parameters can also be changed between multiple runs by 
	using kwargs in the run_sim method.
	'''

	def __init__(self, **kwargs):
		self.h = 1 #time step for ODE solver
		self.N_alleles = 100 #number of alleles
		self.N_iter = 50 #number of evolutionary time steps
		mut = 0.05 #mutation rate

		#Define the mutation matrix
		self._M = np.diag(np.full(self.N_alleles, 1 - mut))
		self._M = self._M + np.diag(np.ones(self.N_alleles - 1)*mut/2, 1)
		self._M = self._M + np.diag(np.ones(self.N_alleles - 1)*mut/2, -1)
		self._M[0,1] = mut
		self._M[self.N_alleles - 1, self.N_alleles - 2] = mut

		#Set initial level of quantitative resistance, as well as resistant 
		#allele and foreign pathgoen introduction times
		self.q_evol = True
		self.q_init = 0.4
		self.R_int = None
		self.Y2_int = None

		#Set parameters and resistance-cost curve
		self._q = np.linspace(0, 1, self.N_alleles)
		self._b = 0.5 + self._q**0.5

		#Set default parameters
		self.transmission = 'fd'
		self.mu = 0.2
		self.gamma = 0.01
		self.r = 0.2
		self.c = 0.2
		self.beta_e = 1
		self.beta_f = 0.7

		#Get kwargs from model initialization and modify parameter values, if
		#h is changed, then _t and N_t must also be changed to compensate
		for key, value in kwargs.items():
			setattr(self, key, value)
		
		self._t = np.arange(0, 750, self.h)
		self.N_t = len(self._t) #number of time steps

	#Differential equation for frequency-dependent transmission
	def dx_fd(self, S, R, Y1, Y2):
		'''
		Calculate the differential of the system dynamics at a particular point 
		using frequency-dependent transmission

		Args:
			S: The number of susceptible indivuals, expressed as a vector of length N_alleles
			R: The number of resistant indivuals, expressed as a vector of length N_alleles
			Y1: The number of individuals infected with the endemic pathogen
			Y2: The number of individuals infected with the foreign pathogen

		Returns:
			dS: The differential for suceptible individuals, vector of length N_alleles
			dR: The differential for resistant individuals, vector of length N_alleles
			dY1: The differential for indiviuals infected with the endemic pathogen
			dY2: The differential for indiviuals infected with the endemic pathogen
		'''
		
		N = np.sum(S) + np.sum(R) + Y1 + Y2
		qS = np.dot(self._q, S)
		qR = np.dot(self._q, R)
			
		dS = S*(self._b - self.mu - self.gamma*N - 
			self._q*(self.beta_e*Y1 + self.beta_f*Y2)/N)
		dR = R*(self._b - self.c - self.mu - self.gamma*N - 
			self._q*(self.r*self.beta_e*Y1 + self.beta_f*Y2)/N)
		dY1 = Y1*(self.beta_e*(qS + self.r*qR)/N - self.mu)
		dY2 = Y2*(self.beta_f*(qS + qR)/N - self.mu)

		return (dS, dR, dY1, dY2)

	#Differential equation for density-dependent transmission
	def dx_dd(self, S, R, Y1, Y2):
		'''
		Calculate the differential of the system dynamics at a particular point 
		using density-dependent transmission

		Args:
			S: The number of susceptible indivuals, expressed as a vector of length N_alleles
			R: The number of resistant indivuals, expressed as a vector of length N_alleles
			Y1: The number of individuals infected with the endemic pathogen
			Y2: The number of individuals infected with the foreign pathogen

		Returns:
			dS: The differential for suceptible individuals, vector of length N_alleles
			dR: The differential for resistant individuals, vector of length N_alleles
			dY1: The differential for indiviuals infected with the endemic pathogen
			dY2: The differential for indiviuals infected with the endemic pathogen
		'''
		
		N = np.sum(S) + np.sum(R) + Y1 + Y2
		qS = np.dot(self._q, S)
		qR = np.dot(self._q, R)

		dS = S*(self._b - self.mu - self.gamma*N - 
			self._q*(self.beta_e*Y1 + self.beta_f*Y2))
		dR = R*(self._b - self.c - self.mu - self.gamma*N - 
			self._q*(self.r*self.beta_e*Y1 + self.beta_f*Y2))
		dY1 = Y1*(self.beta_e*(qS + self.r*qR) - self.mu)
		dY2 = Y2*(self.beta_f*(qS + qR) - self.mu)

		return (dS, dR, dY1, dY2)

	#Run simulation
	def run_sim(self, **kwargs):
		'''
		Run the adaptive dynamics simulation and return the equilibrium 
		abundances at each iteration

		Kwargs:
			All parametes that can be set with class initialization can also be 
			modified during run_sim call

		Returns:
			S_eq: Matrix of shape [N_alleles, N_iter] containing the equilibrium 
				abundances for all susceptible genotypes at each iteration
			R_eq: Matrix of shape [N_alleles, N_iter] containing the equilibrium 
				abundances for all resistant genotypes at each iteration
			Y1_eq: Vector of length N_iter containing the equilibrium number of 
				individuals infected with the endemic pathogen at each iteration
			Y2_eq: Vector of length N_iter containing the equilibrium number of 
				individuals infected with the foreign pathogen at each iteration
		'''
		for key, value in kwargs.items():
			setattr(self, key, value)

		S = np.zeros((self.N_alleles, self.N_t))
		R = np.zeros((self.N_alleles, self.N_t))
		Y1 = np.zeros(self.N_t)
		Y2 = np.zeros(self.N_t)

		#Find matching index to the initial q value
		q_init_ind = (np.abs(self._q - self.q_init)).argmin()

		#Set initial conditions
		S[q_init_ind, 0] = 1
		R[q_init_ind, 0] = 0
		Y1[0] = 1
		Y2[0] = 0
		
		S_eq = np.zeros((self.N_alleles, self.N_iter))
		R_eq = np.zeros((self.N_alleles, self.N_iter))
		Y1_eq = np.zeros(self.N_iter)
		Y2_eq = np.zeros(self.N_iter)

		zero_threshold = 0.01 #Threshold to set abundance values to zero

		for i in range(self.N_iter):
			#Introduce resistant genotype
			if self.R_int != None and i == self.R_int:
				R[np.argmax(S[:,0]), 0] = 1
			
			#Introduce foreign pathogen
			if self.Y2_int != None and i == self.Y2_int:
				Y2[0] = 1

			for j in range(self.N_t - 1):
				#Compute differentials and modify population sizes
				if self.transmission == 'fd':
					diff = self.dx_fd(S[:, j], R[:, j], Y1[j], Y2[j])
				elif self.transmission == 'dd':
					diff = self.dx_dd(S[:, j], R[:, j], Y1[j], Y2[j])
				
				#Append the next time step to the time series
				S[:, j+1] = S[:, j] + self.h*diff[0]
				R[:, j+1] = R[:, j] + self.h*diff[1]
				Y1[j+1] = Y1[j] + self.h*diff[2]
				Y2[j+1] = Y2[j] + self.h*diff[3]

			#Record the final population values at the end of the ecological simulation
			S_eq[:,i] = S[:,-1]
			R_eq[:,i] = R[:,-1]
			Y1_eq[i] = Y1[-1]
			Y2_eq[i] = Y2[-1]

			#Set any population below threshold to 0
			for k in range(self.N_alleles):
				if S[k, -1] < zero_threshold:
					S[k, -1] = 0
				if R[k, -1] < zero_threshold:
					R[k, -1] = 0

			#Assign the values at the end of the ecological simulation to the 
			#first value so the simulation can be re-run
			if self.q_evol:
				#Introduce mutations to q
				S[:,0] = np.dot(self._M, S[:,-1])
				R[:,0] = np.dot(self._M, R[:,-1])

			else:
				#Assign values at the end of the simulation to the begining
				S[:,0] = S[:,-1]
				R[:,0] = R[:,-1]
			
			#Set the initial pathogen abundances to the final abundances from 
			#the previous iteeration
			Y1[0] = Y1[-1]
			Y2[0] = Y2[-1]

		return (S_eq, R_eq, Y1_eq, Y2_eq)

class SimRaster:
	pass 

def classify_sim(S_eq, R_eq):
	'''
	Take simulation outputs and classify the final equilbium as either
	susceptible fixation, resistant fixation, stable polymorphism or cyclic
	polymorphism.

	Args:
		S_eq: Matrix of shape [N_alleles, N_iter] containing the equilibrium 
			abundances for all susceptible genotypes at each iteration
		R_eq: Matrix of shape [N_alleles, N_iter] containing the equilibrium 
			abundances for all resistant genotypes at each iteration	

	Returns:
		designation: numerical identifier for the simulation classification,
			1: susceptible fixation
			2: resistant fixation
			3: stable polymorphism
			4: cyclic polymorphism
		'''

	var_thr = 0.001 #Variance threshold for stable / cyclic polymorphism
	designation = None

	if np.sum(S_eq, axis=0)[-1] > 0 and np.sum(R_eq, axis=0)[-1] == 0:
		designation = 1
	if np.sum(S_eq, axis=0)[-1] == 0 and np.sum(R_eq, axis=0)[-1] > 0:
		designation = 2
	if np.sum(S_eq, axis=0)[-1] > 0 and np.sum(R_eq, axis=0)[-1] > 0:
		if np.var(np.sum(S_eq, axis=0)[-6:-1]) < var_thr:
			designation = 3
		else:
			designation = 4
		
	return designation

def unpack_raster(data, params):
	'''
	Take raw simulation outputs and create rasters of equilibrum dynamics,
	equilibrium general resistance and pathogen proportions.

	Args:
		raster: list of outputs from simulation runs	

	Returns:
		eq: Equilibrium dynamics classifications
		q_map: Equilibrium levels of quantitative resistance
		inf_ratio: Proportion of infected hosts
		y_ratio: Equilibrium proportion of endemic pathogen
		h_ratio: Equilibrium proportion of susceptible and resistant genotypes
	'''

	vars = list(params[0].keys())

	#Get parameter values
	x_vals = np.sort(list(set([param[vars[0]] for param in params])))
	y_vals = np.sort(list(set([param[vars[1]] for param in params])))

	#Get raster dimensions
	n_x = len(x_vals)
	n_y = len(y_vals)

	eq, inf_ratio, y_ratio, h_ratio = [np.zeros((n_x, n_y)) for _ in range(4)]
	q_map = np.zeros((n_x, n_y, 2))

	for i in range(len(data)):
		x_param = params[i][vars[0]]
		y_param = params[i][vars[1]]

		x_ind = np.where(x_vals == x_param)
		y_ind = np.where(y_vals == y_param)

		S_eq, R_eq, Y1_eq, Y2_eq = data[i]

		eq[y_ind, x_ind] = classify_sim(S_eq, R_eq)

		q_map[y_ind, x_ind, 0] = np.argmax(S_eq[:, -1])
		q_map[y_ind, x_ind, 1] = np.argmax(R_eq[:, -1])

		inf_ratio[y_ind, x_ind] = (np.mean(np.sum(S_eq, axis=0)[-6:-1]) + np.mean(np.sum(R_eq, axis=0)[-6:-1])) /  \
			(np.mean(np.sum(S_eq, axis=0)[-6:-1]) + np.mean(np.sum(R_eq, axis=0)[-6:-1]) + np.mean(Y1_eq[-6:-1]) + np.mean(Y2_eq[-6:-1]))

		y_ratio[y_ind, x_ind] = np.mean(Y1_eq[-6:-1]) /  \
			(np.mean(Y1_eq[-6:-1]) + np.mean(Y2_eq[-6:-1]))

		h_ratio[y_ind, x_ind] = np.mean(np.sum(R_eq, axis=0)[-6:-1]) /  \
			(np.mean(np.sum(S_eq, axis=0)[-6:-1]) + np.mean(np.sum(R_eq, axis=0)[-6:-1]))		
			
	return (eq, q_map, inf_ratio, y_ratio, h_ratio)

def unpack_raster_old(raster):
	'''
	Take raw simulation outputs and create rasters of equilibrum dynamics,
	equilibrium general resistance and pathogen proportions.

	Args:
		raster: list of outputs from simulation runs	

	Returns:
		eq: Equilibrium dynamics classifications
		q_map: Equilibrium levels of quantitative resistance
		inf_ratio: Proportion of infected hosts
		y_ratio: Equilibrium proportion of endemic pathogen
		h_ratio: Equilibrium proportion of the resistant genotype
	'''

	#Get raster dimensions
	n_x = len(raster)
	n_y = len(raster[0])

	eq, inf_ratio, y_ratio, h_ratio = [np.zeros((n_x, n_y)) for _ in range(4)]
	q_map = np.zeros((n_x, n_y, 2))

	for i in range(n_x):
		for j in range(n_y):
			S_eq, R_eq, Y1_eq, Y2_eq = raster[i][j]

			eq[i, j] = classify_sim(S_eq, R_eq)

			q_map[i, j, 0] = np.argmax(S_eq[:, -1])
			q_map[i, j, 1] = np.argmax(R_eq[:, -1])

			inf_ratio[i, j] = (np.mean(np.sum(S_eq, axis=0)[-6:-1]) + np.mean(np.sum(R_eq, axis=0)[-6:-1])) /  \
				(np.mean(np.sum(S_eq, axis=0)[-6:-1]) + np.mean(np.sum(R_eq, axis=0)[-6:-1]) + np.mean(Y1_eq[-6:-1]) + np.mean(Y2_eq[-6:-1]))

			y_ratio[i, j] = np.mean(Y2_eq[-6:-1]) /  \
				(np.mean(Y1_eq[-6:-1]) + np.mean(Y2_eq[-6:-1]))

			h_ratio[i, j] = np.mean(np.sum(R_eq, axis=0)[-6:-1]) /  \
				(np.mean(np.sum(S_eq, axis=0)[-6:-1]) + np.mean(np.sum(R_eq, axis=0)[-6:-1]))
			
	return (eq, q_map, inf_ratio, y_ratio, h_ratio)