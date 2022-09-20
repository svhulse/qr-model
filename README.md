# qr-model

Code used to run and analyse simulations for the two locus resistance evolution model

qrmodel.py defines the class Model which can be used to define parameters and run a simulation. Also contains functions classify_sim, which assigns an equiuibrium state to a simulation and unpack_raster, which takes raw simulation raster files and produces rasters of equilibrium states, pathogen and host ratios, and well as overall infected proportions

sim_raster.py is used to run a batch of simulations so that individual parameters can be varried, creating a raster of simulation runs. Contains Model class instances for all simulations used in the paper.

recomb_model.py contains the code used to generate the recombination model and plot its outcome

panels.py contains matplotlib plots for all the plot types used in the paper
