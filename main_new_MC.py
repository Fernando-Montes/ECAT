####################################
##### HANDLING INITIAL IMPORTS #####
####################################

import pickle, os
import numpy as np
from multiprocessing import Pool
from config import (
    z_coordinates, element_names_stripped, viewers,
    fox_file, matrix_directory, pkl_directory
)
from helper_functions import (
    runECAT,
    transmission_histogram,
    plot_cross_section, plot_rays, plot_rays_colored,
    plotInterestingCS,
    checkInitialDistribution
)
from ViewerAnalysis import (
    load_projection_ecat, optimizeMCMC, groupsInfo
)

images2compare = [ ["3", ["1638", "1783"]], ["4", ["1638", "1783"]], ["10", ["1638", "1783"]], ["13", ["1638"]], ["14", ["1638", "1783"]], ["15", ["1638", "1783"]], ["16", ["1638"]] ]
#images2compare = [ ["3", ["1638", "1783"]], ["4", ["1638", "1783"]] ]
numIterations = 1000

def run_iteration(i):
    """Single MC iteration — runs in its own process."""
    np.random.seed()  # random seed issue
    try:
        results = ''

        tuneChanges = [ ['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025], ['Q9', 'B_SC', 0.0949], ['Q11', 'B_SC', -0.0067], ['WF2 Exit', 'dB', -0.00077] ]
        for Q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11']:
            for coord in ['X', 'Y']:
                value = round(np.random.normal(0, 0.00034), 5)
                tuneChanges.append([Q, coord, value])
                results = results + f"{value} "
            value = round(np.random.normal(0, 0.0084), 5)
            tuneChanges.append([Q, 'Roll', value])

        run_cosy_flag = True
        #cosyFile = fox_file
        cosyFile = f"{fox_file}_worker{i}"  # Unique file per worker — prevents collisions
        for groupImages in images2compare:
            group = groupImages[0]
            imagesByGroup = groupImages[1]
            initialDistribution = {
                'type': 'aperture',
                'nRays': 7000,
                'target_alignment' :   {"X": groupsInfo[group]['target_alignment']['X'], "Y": groupsInfo[group]['target_alignment']['Y']+0.1/1000, "R": groupsInfo[group]['target_alignment']['R']},
                'aperture_alignment' : {"X": groupsInfo[group]['pp_alignment']['X'], "Y": groupsInfo[group]['pp_alignment']['Y'], "R": groupsInfo[group]['pp_alignment']['R'], 'separation_distance': 0.48619},
                'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0.0043393, 'sigma' : 10.24/1000},
                'dE' : {'option': 'normal', 'param' : {'mu':  0.020249, 'sigma': 0.077522}},
                'dZ' : {'option': 'fixed',  'param': 0},
            }
            all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes, cosyFile = \
                runECAT( initialDistribution = initialDistribution, run_cosy_flag = run_cosy_flag, foxFile = cosyFile, tuneChanges = tuneChanges,
                    save_rays_flag = False, final_index = viewers[f"{np.max( [int(v) for v in imagesByGroup] )}"]["index"] )
            run_cosy_flag = False

            results = results + load_projection_ecat(imagesByGroup, group, all_x = all_x, all_y = all_y, transmission_indices = transmission_indices,
                plot = False, save_fig = False, all_ax = all_ax, all_dE = all_dE)

        if cosyFile != fox_file:
             os.system(f"rm -rf {matrix_directory}/{cosyFile}")
             os.system(f"rm -rf {pkl_directory}/{cosyFile}.pkl")

        return results

    except Exception as e:
        print(f"[Iteration {i}] FAILED with error: {e}")
        return None  # Signal failure without crashing the pool


if __name__ == "__main__":
    failed = []
    with open("MCoutput.txt", "w") as MCoutput:
        with Pool(processes = 10) as pool:
            for i, iteration_results in enumerate(pool.imap_unordered(run_iteration, range(numIterations))):
                if iteration_results is None:
                    failed.append(i)
                    print(f"[Main] Skipping failed iteration.")
                    continue
                else:
                    MCoutput.write(iteration_results + '\n')
                    MCoutput.flush()

    print(f"\nDone. {numIterations - len(failed)}/{numIterations} iterations succeeded.")
    if failed:
        print(f"Failed iterations: {failed}")
