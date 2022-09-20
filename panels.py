import matplotlib
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

C1 = matplotlib.colors.to_rgb('#196CA9') #S color
C2 = matplotlib.colors.to_rgb('#56AA6D') #R color
C3 = matplotlib.colors.to_rgb('#E07D4F') #Stable PM Color
C4 = matplotlib.colors.to_rgb('#F9C8A7') #Osc PM Color
C5 = matplotlib.colors.to_rgb('#B576CF') #Color for 3B/C
C6 = matplotlib.colors.to_rgb('#83C6C9') #Color for 3B/C
C7 = matplotlib.colors.to_rgb('#D15E5E') #Color for 3B

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def label_axes(ax):
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_yticks([1, 0.75, 0.5, 0.25, 0])
    ax.set_yticklabels(['0', '0.125', '0.25', '0.375', '0.5'])
    ax.set_xticks([1, 0.75, 0.5, 0.25, 0])
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

def eq_plot_1p(eq, ax, legend=True):
    values = np.unique(eq.ravel())
    eq_cmap = matplotlib.colors.ListedColormap([C1,C2,C3])
    bounds = [1,2,3,4]
    eq_norm = matplotlib.colors.BoundaryNorm(bounds, eq_cmap.N)       
    
    im = ax.imshow(eq, extent=[0,1,0,1], cmap=eq_cmap, norm=eq_norm)
    
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[0], label="S Fixation"),
               mpatches.Patch(color=colors[1], label="R Fixation"),
               mpatches.Patch(color=colors[2], label="Polymorphism")]

    label_axes(ax)

    if legend:
        leg = ax.legend(handles = patches, loc=2)
        leg.get_frame().set_linewidth(0.0)

    ax.set_xlabel(r'strength of specific resistance ($r$)')
    ax.set_ylabel(r'cost of specific resistance ($c_r$)')

def eq_plot_2p(eq, ax, legend=True):
    cmap = matplotlib.colors.ListedColormap([C1,C2,C3,C4])
    bounds = [1,2,3,4,5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)                                 
    
    im = ax.imshow(eq, extent=[0,1,0,1], cmap=cmap, norm=norm, interpolation='nearest')    

    values = np.unique(eq.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[0], label=r"$S$ Fixation"),
           mpatches.Patch(color=colors[1], label=r"$R$ Fixation"),
           mpatches.Patch(color=colors[2], label="Stable PM"),
           mpatches.Patch(color=colors[3], label="Cyclic PM")]

    label_axes(ax)
    ax.set_xlabel(r'strength of specific resistance ($r$)')
    ax.set_ylabel(r'cost of specific resistance ($c_r$)')

    if legend:
        leg = ax.legend(handles = patches, bbox_to_anchor=(0,1), frameon=True, ncol=1, loc='upper left')
        leg.get_frame().set_linewidth(0.0)

def q_plot(q_map, ax, legend=True):
    s_q_eq = q_map[:,:,0]
    s_q_eq[s_q_eq == 0] = 99
    r_q_eq = q_map[:,:,1]
    r_q_eq[r_q_eq == 0] = 99

    s_alpha = np.ones((q_map.shape[0], q_map.shape[1]))
    r_alpha = np.ones((q_map.shape[0], q_map.shape[1]))
    s_alpha[s_q_eq == np.max(s_q_eq)] = 0 
    r_alpha[r_q_eq == np.max(r_q_eq)] = 0

    #s_q_max = np.max((99-s_q_eq)/100)
    #r_q_max = np.max((99-r_q_eq)/100)
    s_q_max = 0.4
    r_q_max = 0.4
    s_norm=plt.Normalize(0, s_q_max)
    r_norm=plt.Normalize(0, r_q_max)

    s_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",C2])
    r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",C1])

    ax.imshow((99-r_q_eq)/100, extent=[0,1,0,1], alpha = r_alpha, cmap=s_cmap, norm=r_norm)
    ax.imshow((99-s_q_eq)/100, extent=[0,1,0,1], alpha = s_alpha, cmap=r_cmap, norm=s_norm)

    label_axes(ax)
    
    ax.set_xlabel(r'strength of specific resistance ($r$)')
    ax.set_ylabel(r'cost of specific resistance ($c_r$)')

    ax.annotate(r'$R$: $q_{max} = ' + str(r_q_max) +'$',
        xy=(0.4, 0.1), xycoords='axes fraction',
        xytext=(60, 0), textcoords='offset points', fontsize=SMALL_SIZE,
        arrowprops=dict(arrowstyle='->', facecolor='black'),
        horizontalalignment='center', verticalalignment='center')
    
    ax.annotate(r'$S$: $q_{max} = ' + str(s_q_max) +'$',
        xy=(0.65, 0.47), xycoords='axes fraction',
        xytext=(0, -40), textcoords='offset points', fontsize=SMALL_SIZE,
        arrowprops=dict(arrowstyle='->', facecolor='black'),
        horizontalalignment='center', verticalalignment='center')

    if legend:
        patches = [mpatches.Patch(color=C1, label=r"General Resistance in $S$"),
            mpatches.Patch(color=C2, label=r"General Resistance in $R$")]
        leg = ax.legend(handles = patches, bbox_to_anchor=(0,1), frameon=True, ncol=1, loc='upper left')
        leg.get_frame().set_linewidth(0.0)

def q_plot_2p(q_eq, ax):
    q_eq[q_eq == 0] = 99
    q_norm = plt.Normalize(0, np.max((99-q_eq)/100))
    q_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",C7])
    
    q_im = ax.imshow((99-q_eq)/100, extent=[0,1,0,1], cmap=q_cmap, norm=q_norm)

    label_axes(ax)

    box = ax.get_position()
    #axColor = plt.axes([box.x0*1.01 + box.width * 1.05, box.y0, 0.01, box.height])
    #c = plt.colorbar(q_im, cax=axColor)
    axColor = plt.axes([box.x0, box.y0 - 0.075, box.width, 0.01])
    plt.colorbar(q_im, axColor, orientation='horizontal')
    #c.set_ticks([0, 0.08])
    #c.ax.tick_params(length=0)
    #c.ax.set_yticklabels(['no resistance', 'maximal resistance'], rotation=90, va='bottom', fontsize=SMALL_SIZE)

    ax.set_xlabel(r'strength of specific resistance ($r$)')
    ax.set_ylabel(r'cost of specific resistance ($c_r$)')

    return q_im

def prev_plot_1p(inf_ratio, ax):
    norm = plt.Normalize(np.min(1-inf_ratio),np.max(1-inf_ratio))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",C3])
    q_im = ax.imshow(1-inf_ratio, extent=[0,1,0,1], norm=norm, cmap=cmap)
    
    label_axes(ax)
    
    ax.set_xlabel(r'strength of specific resistance ($r$)')
    ax.set_ylabel(r'cost of specific resistance ($c_r$)')
    ax.set_title(r'Prevalence of $I_e$', fontsize=BIGGER_SIZE, pad=10)
    
    box = ax.get_position()
    axColor = plt.axes([box.x0*1.01 + box.width * 1.05, box.y0, 0.01, box.height])
    plt.colorbar(q_im, cax=axColor)

def y_ratio_plot(y_ratio, ax):
    norm = plt.Normalize(0,np.max(y_ratio))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [C5,"white"])
    
    y_im = ax.imshow(y_ratio, extent=[0,1,0,1], cmap=cmap, norm=norm)
    
    label_axes(ax)
    
    box = ax.get_position()
    axColor = plt.axes([box.x0*1.01 + box.width * 1.05, box.y0, 0.01, box.height])

    b = plt.colorbar(y_im, cax=axColor)
    b.set_ticks([0, np.max(y_ratio)*0.77])
    b.ax.tick_params(length=0)
    b.ax.set_yticklabels(['foreign', 'endemic'], rotation=90, va='bottom', fontsize=SMALL_SIZE)

    ax.set_xlabel(r'strength of specific resistance ($r$)')
    ax.set_ylabel(r'cost of specific resistance ($c_r$)')

def h_ratio_plot(h_ratio, ax):
    norm = plt.Normalize(0,np.max(h_ratio))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [C5,C6])
    
    y_im = ax.imshow(h_ratio, extent=[0,1,0,1], cmap=cmap, norm=norm)
    
    label_axes(ax)
    
    box = ax.get_position()
    axColor = plt.axes([box.x0*1.01 + box.width * 1.05, box.y0, 0.01, box.height])
    b = plt.colorbar(y_im, cax=axColor)
    b.set_ticks([0, 0.82])
    b.ax.tick_params(length=0)
    b.ax.set_yticklabels(['resistant', 'susceptible'], rotation=90, va='bottom', fontsize=SMALL_SIZE)

def q_evol_plot(model, ax):
    s, r, y1, y2 = model.run_sim()
    y_q_labels = ['1', '0.75', '0.5', '0.25', '0']
    x_q_labels = ['0', '25', '50', '75', '100']

    s_norm=plt.Normalize(0,np.max(s)/5)
    s_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",C1])
    
    ax.imshow(s, cmap = s_cmap, norm = s_norm)
    ax.set_yticks([0,25,50,75,100])
    ax.set_xticks([0,25,50,75,100])
    ax.set_yticklabels(y_q_labels)
    ax.set_xticklabels(x_q_labels)
    ax.set_ylabel(r'strength of general resistance ($q$)')
    ax.set_xlabel(r'evolutionary time step')
