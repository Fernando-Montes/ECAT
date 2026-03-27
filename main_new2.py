####################################
##### HANDLING INITIAL IMPORTS #####
####################################

# Import necessary packages
import pickle, os
import numpy as np

# Import system configuration [See "config.py"]
from config import (
    z_coordinates, element_names_stripped, viewers,                          # Defines system geometry
    fox_file, matrix_directory, pkl_directory
)

# Import helper functions for processing and plotting [See "helper_functions.py"]
from helper_functions import (
    runECAT,                                                                 # Run ECAT simulation
    transmission_histogram,                                                  # Transmission analysis
    plot_cross_section, plot_rays, plot_rays_colored,                        # Additional plotting
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
#
# #Run ECAT
# # initialDistribution = { 'type': 'mixed_rays',
# #                         'nRays': 500,
# #                         'target' : {'option': 'circle', 'param': {'center': (0, 0), 'radius': 1.5/1000}}, \
# #                         'angles' : {'option': 'circle', 'param': {'center': (0, 0/1000), 'radius': 1.0 / 1000}}, \
# #                         'dE' : {'option' : 'uniform',  'param' : {'min': -0.2/100, 'max': 0.2/100}}, \
# #                         'dZ' : {'option': 'fixed',  'param': 0}, }
# # initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
# #                         'nRays': 50,
# #                         'target_alignment' :   {"X": 0.03055/1000, "Y": -0.01099/1000+0.1/1000, "R": 0.75/1000},
# #                         'aperture_alignment' : {"X": 0.311246954/1000, "Y": 8.5930652/1000, "R": 0.5/1000, 'separation_distance': 0.48619},
# #                         'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0.0043393, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
# #                         'dE' : {'option': 'normal', 'param' : {'mu':  0.020249, 'sigma': 0.077522}},      # 2, 8
# #                         'dZ' : {'option': 'fixed',  'param': 0}, }
#
# initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
#                         'nRays': 50,
#                         'target_alignment' :   {"X": 0.03055 / 1000, "Y": -0.01099 / 1000+0.1/1000, "R": 0.75 / 1000},
#                         'aperture_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, 'separation_distance': 0.48619},
#                         'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0.0043393, 'sigma' : 10.24/1000},
#                         #'dE' : {'option': 'normal', 'param' : {'mu':  0.020249, 'sigma': 0.077522}},
#                         'dE' :  {'option' : 'stepped',  'param' : {'min': -0.03, 'max': 0.03, 'steps': 5}},
#                         'dZ' : {'option': 'fixed',  'param': 0}, }
# #tuneChanges = [ ['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025], ['Q9', 'B_SC', 0.0949], ['Q11', 'B_SC', -0.0067], ['WF2 Exit', 'dB', -0.00077] ]
# tuneChanges = [ ['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025], ['Q9', 'X', 0.000834] ]
# #tuneChanges = [ ['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025] ]
# #tuneChanges = [ ]
# all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes, cosyFile = \
#     runECAT(initialDistribution = initialDistribution, run_cosy_flag = True, tuneChanges = tuneChanges,
#         save_rays_flag = False, final_index = element_names_stripped.index('VD1879') )
#
# # Plots initial ray distribution used (uses saved rays_file)
# #checkInitialDistribution(pepperpot = True)
#
# # Histogram to show the percentage of rays stopped at a given position
# transmission_histogram(transmitted_x, element_names_stripped, transmission_indices,
#     focal_planes = [ ("FP1 Slits", "VD1542", "FP 1"), ("FP2 Slits", "VD1638", "FP 2"), ("FP3 Slits", "FP3 Slits", "FP 3"),
#         ("FP4 Slits", "DSSD", "FP 4") ] )
#
# # Plots x-vs-z and y-vs-z to trace rays (all or only up to transmitted points) along the beamline
# #plot_rays(transmitted_x, transmitted_y, z_coordinates, element_names_stripped)
# color_dict = {
#     "color_list": [dE[0] for dE in all_dE],
#     "n_bins": 5,
#     "color_key": "Initial dE/E"
# }
# plot_rays_colored(transmitted_x, transmitted_y, z_coordinates, color_dict, element_names_stripped)
#
#
# ## Plots beamline cross section at a given location
# # plot_cross_section(element_names_stripped.index('VD1836'), beampipes, chamber_names, element_names_stripped, all_x, all_y, transmission_indices, z_coordinates)
# #
# # Plots multiple beamline cross sections at the given locations
# # displaySec = ['Target Center', 'FC1485', 'VD1515', 'VD1542', 'AP1568 Aperture Drive', 'VD1638', 'VD1688', \
# #                                       'FP3 Slits & VD1783', 'VD1836', 'VD1879']
# # displaySec = ['VD1836']
# # plotInterestingCS(displaySec, element_names_stripped, all_x, all_y, transmission_indices, saveFile = 'test.pdf')
# #
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

# #images2compare = [ ["7", ["1515", "1542", "1638"]], ["8", ["1515", "1542", "1638"]], ["9", ["1542", "1638"]], ["10", ["1515", "1542", "1638"]], ["13", ["1638"]], ["14", ["1515", "1542", "1638"]], ["15", ["1515", "1542", "1638"]], ["16", ["1515"]] ]
# images2compare = [ ["7", ["1638"]], ["8", ["1638"]], ["9", ["1638"]], ["10", ["1638"]], ["13", ["1638"]], ["14", ["1638"]], ["15", ["1638"]] ]
# #images2compare = [ ["7", ["1515", "1542", "1638", "1688", "1783"]], ["8", ["1688", "1783"]],  ["9", ["1688", "1783"]], ["10", ["1783"]], ["14", ["1688", "1783"]],  ["15", ["1688", "1783"]] ]
# #images2compare = [ ["7", ["1688", "1783"]],  ["10", ["1783"]] ]
# #images2compare = [ ["7", ["1638", "1688", "1783"]], ["10", ["1638", "1783"]] ]
# images2compare = [ ["3", ["1638"]], ["4", ["1638"]], ["10", ["1638"]] ]
#images2compare = [ ["5", ["1638"]], ["7", ["1638"]], ["8", ["1638"]], ["9", ["1638"]], ["6", ["1638"]], ["3", ["1638"]], ["4", ["1638"]], ["10", ["1638"]], ["13", ["1638"]], ["14", ["1638"]], ["15", ["1638"]], ["16", ["1638"]] ]
#images2compare = [ ["10", ["1638"]], ["13", ["1638"]], ["14", ["1638"]], ["15", ["1638"]], ["16", ["1638"]] ]
#images2compare = [ ["5", ["1783"]], ["7", ["1783"]], ["10", ["1783"]], ["14", ["1783"]], ["15", ["1783"]] ]
images2compare = [ ["3", ["1783"]], ["4", ["1783"]], ["5", ["1783"]], ["6", ["1783"]], ["7", ["1783"]], ["8", ["1783"]], ["9", ["1783"]], ["10", ["1783"]], ["14", ["1783"]], ["15", ["1783"]] ]
#images2compare = [ ["3", ["1688"]], ["4", ["1688"]], ["8", ["1688"]], ["9", ["1688"]], ["14", ["1688"]], ["15", ["1688"]] ]
#images2compare = [ ["18", ["1638", "1688", "1783"]], ["19", ["1638", "1688", "1783"]] ]
#images2compare = [ ["18", ["1515", "1542", "1638", "1783"]] ]
#images2compare = [ ["3", ["1836", "1879"]], ["4", ["1836", "1879"]], ["5", ["1836", "1879"]], ["6", ["1836", "1879"]], ["7", ["1836", "1879"]], ["8", ["1836", "1879"]], ["9", ["1836", "1879"]], ["10", ["1879"]], ["13", ["1879"]], ["15", ["1836", "1879"]] ]
#images2compare = [ ["10", ["1879"]], ["13", ["1879"]], ["15", ["1879"]], ["2", ["1879"]], ["3", ["1879"]], ["4", ["1879"]] ]
#images2compare = [ ["2", ["1836"]], ["3", ["1836"]], ["4", ["1836"]], ["7", ["1836"]], ["8", ["1836"]], ["9", ["1836"]], ["15", ["1836"]], ["16", ["1836"]]]
#images2compare = [ ["4", ["1836", "1879"]] ]
#images2compare = [ ["5", ["1783"]], ["7", ["1783"]], ["8", ["1783"]], ["9", ["1783"]] ]

#
# tuneChanges = None
# # tuneChanges = [ ['Q1', 'Y', -0.000410], ['Q1', 'B_SC', 0.005898], ['Q2', 'Y', -0.00055386], ['Q2', 'B_SC', -0.017934],  ['B2 Exit', 'dB', -0.0049416] ]
# # tuneChanges = [ ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['WF1 Exit', 'dB', -0.0125] ]   # +0.0016 WF1
# # tuneChanges = [['Q1', 'Y', -0.0004005534288656388], ['Q1', 'B_SC', 0.01016527258784548], ['Q2', 'Y', -0.0005769743274796191], ['Q2', 'B_SC', -0.006159636853166115 ], ['Q3', 'B_SC', 0.00036907811424478215], ['Q4', 'B_SC', -0.013252101342486288], ['Q5', 'B_SC', 0.006975071308727706], ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['Q6', 'B_SC', -0.026865801259426076], ['Q7', 'B_SC', -0.006350239818653578], ['WF1 Exit', 'dB', -0.0125], \
# #                 ['B5 Exit', 'dB', -0.005], ['B6 Exit', 'dB', -0.005], ['WF2 Exit', 'dB', -0.005]]
# # tuneChanges = [['Q1', 'Y', -0.0004005534288656388], ['Q1', 'B_SC', 0.01016527258784548], ['Q2', 'Y', -0.0005769743274796191], ['Q2', 'B_SC', -0.006159636853166115 ], ['Q3', 'B_SC', 0.00036907811424478215], ['Q4', 'B_SC', -0.013252101342486288], ['Q5', 'B_SC', 0.006975071308727706], ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['Q6', 'B_SC', -0.026865801259426076], ['Q7', 'B_SC', -0.006350239818653578], ['WF1 Exit', 'dB', -0.0125], ['B5 Exit', 'dB', -0.0053], ['B6 Exit', 'dB', -0.0053], ['WF2 Exit', 'dB', -0.0059], ['Q9', 'X', -0.0005]]
#tuneChanges = [ ['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025], ['Q9', 'B_SC', 0.0949], ['Q11', 'B_SC', -0.0067], ['WF2 Exit', 'dB', -0.00077] ]
tuneChanges = [ ['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025], ['Q9', 'X', -0.000834] ]
#tuneChanges = [ ['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025] ]
run_cosy_flag = True
#
#run_cosy_flag = False
metric = 0
cosyFile = fox_file
for groupImages in images2compare:
    group = groupImages[0]
    imagesByGroup = groupImages[1]
    initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
                            'nRays': 7000,
                            'target_alignment' :   {"X": groupsInfo[group]['target_alignment']['X'], "Y": groupsInfo[group]['target_alignment']['Y']+0.1/1000, "R": groupsInfo[group]['target_alignment']['R']},
                            'aperture_alignment' : {"X": groupsInfo[group]['pp_alignment']['X'], "Y": groupsInfo[group]['pp_alignment']['Y'], "R": groupsInfo[group]['pp_alignment']['R'], 'separation_distance': 0.48619},
                            'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0.0043393, 'sigma' : 10.24/1000},
                            #'angles' : {'mu_x' : -0.12/1000, 'mu_y' : -0.0027, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
                            'dE' : {'option': 'normal', 'param' : {'mu':  0.020249, 'sigma': 0.077522}},
                            'dZ' : {'option': 'fixed',  'param': 0}, }
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes, cosyFile = \
        runECAT( initialDistribution = initialDistribution, run_cosy_flag = run_cosy_flag, foxFile = cosyFile, tuneChanges = tuneChanges,
            save_rays_flag = False, final_index = viewers[f"{np.max( [int(v) for v in imagesByGroup] )}"]["index"] )
    run_cosy_flag = False # Only needs to run cosy once
    metric = metric + load_projection(imagesByGroup, group, all_x = all_x, all_y = all_y, transmission_indices = transmission_indices,
        plot = True, metric = 'chisq_1d', save_fig = True, all_ax = all_ax, all_dE = all_dE)

# # if cosyFile != fox_file:
# #     os.system(f"rm -rf {matrix_directory}/{cosyFile}")
# #     os.system(f"rm -rf {pkl_directory}/{cosyFile}.pkl")
# # print( metric )

# # import multiprocessing as mp
# # if __name__ == "__main__":
# #     try:
# #         mp.set_start_method("spawn")
# #     except RuntimeError:
# #         pass
# #     params = [
# #         {'elem': 'initialDist', 'par': 'mu_y'},
# #         {'elem': 'initialDist', 'par': 'mu'},
# #         {'elem': 'initialDist', 'par': 'sigma'},
# #         {'elem': 'Q1',          'par': 'Y'},
# #         {'elem': 'Q1',          'par': 'B_SC'},
# #         {'elem': 'Q2',          'par': 'Y'},
# #         {'elem': 'Q2',          'par': 'B_SC'},
# #         #{'elem': '1515',        'par': 'dist'},
# #         #{'elem': 'B1',          'par': 'Y'},
# #         #{'elem': 'B1 Exit',     'par': 'dB'},
# #         # {'elem': 'B2',          'par': 'Y'},
# #         {'elem': 'B2 Exit',     'par': 'dB'},
# #         {'elem': 'Q3',          'par': 'B_SC'},
# #         {'elem': 'Q4',          'par': 'B_SC'},
# #         {'elem': 'Q5',          'par': 'B_SC'},
# #         {'elem': 'B3 Exit',     'par': 'dB'},
# #         {'elem': 'B4 Exit',     'par': 'dB'},
# #         {'elem': 'Q6',          'par': 'B_SC'},
# #         {'elem': 'Q7',          'par': 'B_SC'},
# #         {'elem': 'WF1 Exit',    'par': 'dB'}
# #     ]
# #     # Viewers to be used in the optimization: array of [group, viewer number]
# #     #images2compare = [ ["7", ["1515", "1542"]], ["8", ["1515", "1542"]], ["9", ["1542"]], ["10", ["1515", "1542"]], ["14", ["1515", "1542"]], ["15", ["1515", "1542"]], ["16", ["1515"]] ] #, ["17", ["1515"]] ]
# #     #images2compare = [ ["7", ["1638"]], ["8", ["1638"]], ["9", ["1638"]], ["10", ["1638"]], ["13", ["1638"]], ["14", ["1638"]], ["15", ["1638"]] ]
# #     images2compare = [ ["7", ["1515", "1542", "1638"]], ["8", ["1515", "1542", "1638"]], ["9", ["1542", "1638"]], ["10", ["1515", "1542", "1638"]], ["13", ["1638"]], ["14", ["1515", "1542", "1638"]], ["15", ["1515", "1542", "1638"]], ["16", ["1515"]] ]
# #     optimizeMCMC(params, images2compare, fresh_start = False)
