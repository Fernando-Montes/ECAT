####################################
##### HANDLING INITIAL IMPORTS #####
####################################

# Import necessary packages
import pickle, os
import numpy as np

# Import system configuration [See "config.py"]
from config import (
    z_coordinates, element_names_stripped, viewers           # Defines system geometry
)

# Import helper functions for processing and plotting [See "helper_functions.py"]
from helper_functions import (
    runECAT,                                                                 # Run ECAT simulation
    transmission_histogram,                                                  # Transmission analysis
    plot_cross_section, plot_rays,                                           # Additional plotting
    plotInterestingCS,                                                       # Additional plotting
    checkInitialDistribution
)

####################################
########## BASIC ECAT RUN ##########
####################################
'''
runECAT : runs simulation
Comment/uncomment plotting as needed
'''

# # Run ECAT
# initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
#                         'nRays': 5000,
#                         'target_alignment' :   {"X": 0.03055/1000, "Y": -0.01099/1000, "R": 0.75/1000},
#                         'aperture_alignment' : {"X": -0.448502/1000, "Y": -0.449370/1000, "R": 10.46/1000, 'separation_distance': 0.48619},
#                         'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0/1000, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
#                         'dE' : {'option': 'normal', 'param' : {'mu':  2/100, 'sigma': 8/100}},      # 2, 8
#                         'dZ' : {'option': 'fixed',  'param': 0}, }
# tuneChanges = [ ['Q1', 'B_SC', -0.0005], ['Q2', 'B_SC', 0.00001], ['B2 Exit', 'dB', -0.0001] ]   # dB and B_SC in fraction: 0.01 means 1%
# all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes = \
#     runECAT(initialDistribution = initialDistribution, run_cosy_flag = False, tuneChanges = None,
#         save_rays_flag = False, final_index = element_names_stripped.index('DSSD') )

# # Plots initial ray distribution used (uses saved rays_file)
# checkInitialDistribution(pepperpot = True)
#
# # Histogram to show the percentage of rays stopped at a given position
# transmission_histogram(transmitted_x, element_names_stripped, transmission_indices,
#     focal_planes = [ ("FP1 Slits", "VD1542", "FP 1"), ("FP2 Slits", "VD1638", "FP 2"), ("FP3 Slits", "FP3 Slits", "FP 3"),
#         ("FP4 Slits", "DSSD", "FP 4") ] )
#
# # Plots x-vs-z and y-vs-z to trace rays (all or only up to transmitted points) along the beamline
# plot_rays(transmitted_x, transmitted_y, z_coordinates, element_names_stripped)
#
# # Plots beamline cross section at a given location
# plot_cross_section(element_names_stripped.index('VD1515'), beampipes, chamber_names, element_names_stripped, all_x, all_y, transmission_indices, z_coordinates)
#
# # Plots multiple beamline cross sections at the given locations
# displaySec = ['Target Center', 'FC1485', 'VD1515', 'VD1542', 'AP1568 Aperture Drive', 'VD1638', 'VD1688', \
#                                       'FP3 Slits & VD1783', 'VD1836', 'VD1879']
# plotInterestingCS(displaySec, element_names_stripped, all_x, all_y, transmission_indices)

##############################
###### Viewer Analysis #######
##############################
'''
optimizeMCMC : runs optimization
Comment/uncomment code as needed
'''

from ViewerAnalysis import (
    load_projection, optimizeMCMC, groupsInfo
)

images2compare = [ ["7", ["1515", "1542"]], ["10", ["1515", "1542"]] ]
metric = 0
for groupImages in images2compare:
    group = groupImages[0]
    imagesByGroup = groupImages[1]
    initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
                            'nRays': 2000,
                            'target_alignment' :   {"X": groupsInfo[group]['target_alignment']['X'], "Y": groupsInfo[group]['target_alignment']['Y'], "R": groupsInfo[group]['target_alignment']['R']},
                            'aperture_alignment' : {"X": groupsInfo[group]['pp_alignment']['X'], "Y": groupsInfo[group]['pp_alignment']['Y'], "R": groupsInfo[group]['pp_alignment']['R'], 'separation_distance': 0.48619},
                            'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0/1000, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
                            'dE' : {'option': 'normal', 'param' : {'mu':  2/100, 'sigma': 8/100}},      # 2, 8
                            'dZ' : {'option': 'fixed',  'param': 0}, }
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes = \
        runECAT( initialDistribution = initialDistribution, run_cosy_flag = False, tuneChanges = None,
            save_rays_flag = False, final_index = viewers[f"{np.max( [int(v) for v in imagesByGroup] )}"]["index"] )
    metric = metric + load_projection(imagesByGroup, group, all_x = all_x, all_y = all_y, transmission_indices = transmission_indices,
        plot = True, metric = 'chisq_1d')
print( metric )

# import multiprocessing as mp
# if __name__ == "__main__":
#     try:
#         mp.set_start_method("spawn")
#     except RuntimeError:
#         pass
#     params = [
#         {'elem': 'initialDist', 'par': 'mu_y'},
#         #{'elem': '1515',        'par': 'dist'},
#         {'elem': 'B1 Exit',     'par': 'dB'},
#         {'elem': 'B2 Exit',     'par': 'dB'},
#         #{'elem': 'B3 Exit',     'par': 'dB'},
#         #{'elem': 'B4 Exit',     'par': 'dB'},
#         #{'elem': 'WF1 Exit',    'par': 'dB'}
#     ]
#     # Viewers to be used in the optimization: array of [group, viewer number]
#     images2compare = [ ["7", ["1515", "1542"]] ]
#     optimizeMCMC(params, images2compare, fresh_start = False)
