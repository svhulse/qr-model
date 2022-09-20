import numpy as np
import matplotlib

from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Helvetica Neue"
matplotlib.rcParams['figure.dpi'] = 300

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

h = 0.1 #time step for ODE solver
t = np.arange(0, 4000, h)
N_t = len(t) #number of time steps

beta_e = 1 #transmission rate of endemic pathogen
beta_f = 0.7 #transmission rate of foreign pathogen
q = 0.84 #equilibrium level of quantitative resistance
r = 0.2 #qualitative resistance level

b = 1.5 #background birth rate
c_q = 1 - q**0.5 #cost of quantitative resistance
c_r = 0.1 #cost of qualitative resistance

costs = np.array([c_q + c_r, c_q, c_r, 0])
inf_e = beta_e * np.array([q*r, q, r, 1])
inf_f = beta_f * np.array([q, q, 1, 1])
inf = np.transpose(np.matrix((inf_e, inf_f)))

gamma = 0.001 #coefficient of density dependence
mu = 0.2 #death rate

def df(s, i, p):
  N = np.sum(s) + np.sum(i)
  N_allele = np.sum(s)
    
  Q_freq = (s[0] + s[1]) / N_allele
  q_freq = (s[2] + s[3]) / N_allele
  R_freq = (s[0] + s[2]) / N_allele
  r_freq = (s[1] + s[3]) / N_allele

  mut = np.array([Q_freq*R_freq, Q_freq*r_freq, q_freq*R_freq, q_freq*r_freq])

  ds = np.multiply((1-p), np.multiply(s,b) - costs - gamma*N - np.dot(inf, i)/N - mu) + p*mut*np.sum(s)
  di = np.multiply(i, np.dot(s, inf)/N - mu)
  hrrg = mut

  return (ds, di, hrrg)

def run_sim(p):
    S = np.zeros((4, N_t)) #S[0,:] = QR, S[1,:] = Qr, S[2,:] = qR, S[3,:] = qr
    I = np.zeros((2, N_t)) #I[0,:] = endemic, I[1,:] = foreign

    #Set initial conditions
    S[:,0] = [0,6.422,7.039,0]
    I[:,0] = [21.185,29.461]

    for i in range(N_t - 1):
      #Compute differentials and modify population sizes
      diff = df(S[:,i], I[:,i], p)
        
      S[:, i+1] = S[:, i] + h*diff[0]
      I[:, i+1] = I[:, i] + h*diff[1]

    return (S, I)
  
fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,4))

S0, I0 = run_sim(0)

ax[0].plot(t, S0[1,:], label = r'$Q^+S$')
ax[0].plot(t, S0[2,:], label = r'$Q^-R$')
ax[0].plot(t, S0[0,:], label = r'$Q^+R$')
ax[0].plot(t, S0[3,:], label = r'$Q^-S$')
ax[0].legend(frameon=False)
ax[0].set_ylabel(r'abundance')
ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$p = 0$', fontsize=MEDIUM_SIZE, pad=10)

S1, I1 = run_sim(0.005)

ax[1].plot(t, S1[1,:], label = r'$Q^+S$')
ax[1].plot(t, S1[2,:], label = r'$Q^-R$')
ax[1].plot(t, S1[0,:], label = r'$Q^+R$')
ax[1].plot(t, S1[3,:], label = r'$Q^-S$')
ax[1].legend(frameon=False)
ax[1].set_ylabel(r'abundance')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$p = 0.005$', fontsize=MEDIUM_SIZE, pad=10)

S2,I2 = run_sim(0.1)

ax[2].plot(t, S2[1,:], label = r'$Q^+S$')
ax[2].plot(t, S2[2,:], label = r'$Q^-R$')
ax[2].plot(t, S2[0,:], label = r'$Q^+R$')
ax[2].plot(t, S2[3,:], label = r'$Q^-S$')
ax[2].legend(frameon=False)
ax[2].set_ylabel(r'abundance')
ax[2].set_xlabel(r'$t$')
ax[2].set_title(r'$p = 0.1$', fontsize=MEDIUM_SIZE, pad=10)

ax[0].annotate("A", xy=(-0.15, 1.1), xycoords="axes fraction", fontsize=BIGGER_SIZE)
ax[1].annotate("B", xy=(-0.15, 1.1), xycoords="axes fraction", fontsize=BIGGER_SIZE)
ax[2].annotate("C", xy=(-0.15, 1.1), xycoords="axes fraction", fontsize=BIGGER_SIZE)


fig.savefig('/home/sam/Projects/QR_Model/Figures/recomb_plot2.svg', bbox_inches='tight', pad_inches=0.1)
