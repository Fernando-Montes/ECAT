import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from emcee.backends import HDFBackend
import multiprocessing as mp
from functools import partial
import corner

from config import viewers, element_names_stripped, results_directory
from helper_functions import runECAT

groupsInfo = {
    '1': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '2': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '3': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-21, -7),  "label": "FP1 Slits: -21 to -7 mm"},
        },
    '4': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.07342537 / 1000, "Y": -0.06722703 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (7, 21),    "label": "FP1 Slits: 7 to 21 mm"},
        },
    '5': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '6': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) 0% dE/E"},
        'fp1_slits' : {"slits_mm": (-50, 50),  "label": "FP1 Slits: ±50 mm"},
        },
    '7': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": [-7, 7], "label": "FP1 Slits: ±7 mm"},
        },
    '8': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-21, -7),  "label": "FP1 Slits: -21 to -7 mm"},
        },
    '9': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.448502 / 1000, "Y": -0.449370 / 1000, "R": 10.46 / 1000, "label": "20 mrad (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (7, 21),    "label": "FP1 Slits: 7 to 21 mm"},
        },
    '10': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.1284349631 / 1000, "Y": 0.1781644397 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '11': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -0.1284349631 / 1000, "Y": 0.1781644397 / 1000, "R": 0.5 / 1000, "label": "1 MM (0 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '12': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -8.068629955 / 1000, "Y": 0.030415035 / 1000, "R": 0.5 / 1000, "label": "1 MM (+15 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '13': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -4.239219775 / 1000, "Y": 0.431691050 / 1000, "R": 0.5 / 1000, "label": "1 MM (+7.5 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '14': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": 8.061907576 / 1000, "Y": -0.310614354 / 1000, "R": 0.5 / 1000, "label": "1 MM (-15 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '15': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": 4.029862244 / 1000, "Y": -0.504692800 / 1000, "R": 0.5 / 1000, "label": "1 MM (-7.2 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '16': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": -10.62294395 / 1000, "Y": -0.18692655 / 1000, "R": 0.5 / 1000, "label": "1 MM (+20 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    '17': {
        'target_alignment' : {"X": 0.03055 / 1000, "Y": -0.01099 / 1000, "R": 0.75 / 1000},
        'pp_alignment' : {"X": 10.30665605 / 1000, "Y": -0.18692655 / 1000, "R": 0.5 / 1000, "label": "1 MM (-20 mrad Position) -2% dE/E"},
        'fp1_slits' : {"slits_mm": (-7, 7),    "label": "FP1 Slits: ±7 mm"},
        },
    }

# Dictionary that sets the max. variance a given type of parameter can possibly have (to be used as bounds in the
# parameter optimization
bounds = {
    'mu_y' : [-0.006, 0.006],  # +- 6mrad
    'Roll' : [-0.003, 0.003],  # +-1.1deg rotation around beam axis
    'dXY'  : [-0.001, 0.001],  # +-1mm
    'B_SC' : [-0.003, 0.003],  # +-0.3% field
    'dist' : [-0.003, 0.003],  # +-3mm
    'dB'   : [-0.011, 0.011],  # Possible miscalibration +-1.1% according to 2024 tuning
}

# Trouble shooting... can be deleted
def load_2d_image(gorup_number, viewer, image_path, plot = True, save_path = None):

    data = np.load(image_path)
    corrected_full_image = data["corrected_full_image"]
    xf_range = data["xf_range"]
    yf_range = data["yf_range"]

    if plot:
        plt.imshow(
            corrected_full_image,
            extent = (xf_range[0], xf_range[-1], yf_range[-1], yf_range[0]),
            cmap = 'viridis', aspect = 'auto'
        )
        plt.colorbar()

        plt.xlim(xf_range[0], xf_range[-1])
        plt.ylim(yf_range[-1], yf_range[0])
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')

        # Title depending on POV
        if "_cosy" in image_path:
            pov = "Looking at Beam / COSY POV"
        else:
            pov = "Looking with Beam / Viola POV"

        plt.title(f"Group {gorup_number} - {viewer}\n({pov})", fontweight = 'bold')

        if save_path:
            plt.savefig(save_path, dpi = 300)
            plt.close()
        else:
            plt.show()
            plt.close()

    return corrected_full_image, xf_range, yf_range

# Loading projections from csv files AND/OR returns metric value when comparing viewer and ECAT simulations
def load_projection(imagesByGroup, group_number, all_x = None, all_y = None, transmission_indices = None, plot = True, metric = None, save_fig = False):
    '''
    metric : metric to compare viewer images against ECAT results. Can be: 'chisq_1d (X and Y projections of a viewer), chisq_2d (viewer)'
    returns metric value
    '''
    chisq_x, chisq_y = 0, 0
    metric_value = 0
    group_path = f"PartD_ViewerAnalysis/Group{group_number}"
    for viewer in imagesByGroup:   # go viewer by viewer
        if metric == 'chisq_1d' or plot == True:
            x_projection_path = os.path.join(group_path, f"VD{viewer}_Group{group_number}_xf_cosy.csv")
            y_projection_path = os.path.join(group_path, f"VD{viewer}_Group{group_number}_yf_cosy.csv")
            # Load 1d viewer projections and skip any lines starting with a '#'
            df_x = pd.read_csv(x_projection_path, comment = "#")
            df_y = pd.read_csv(y_projection_path, comment = "#")

            # Extract the 1d viewer projections data
            x_pos = -df_x.iloc[:, 0].to_numpy()
            x_intensity = df_x.iloc[:, 1].to_numpy()
            y_pos = df_y.iloc[:, 0].to_numpy()
            y_intensity = df_y.iloc[:, 1].to_numpy()

            # Creating 1d viewer histograms (binned)
            x_bins = 50
            x_scale = x_bins/len(x_pos)
            x_intensityC = [i*x_scale for i in x_intensity]
            x_intensityC_sum = np.sum(x_intensityC)
            x_intensityC = [max(1, round(100*x_bins*i/x_intensityC_sum)) for i in x_intensityC]   # Scale binned viewer so sum is 100*x_bins
            x_intensity  = [100*x_bins*i/x_intensityC_sum for i in x_intensity]                   # Scale orig viewer so agrees with binned
            x_vd = [ p for p,n in zip(x_pos, x_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed

            y_bins = 50
            y_scale = y_bins/len(y_pos)
            y_intensityC = [i*y_scale for i in y_intensity]
            y_intensityC_sum = np.sum(y_intensityC)
            y_intensityC = [max(1, round(100*y_bins*i/y_intensityC_sum)) for i in y_intensityC]   # Scale binned viewer so sum is 100*y_bins
            y_intensity  = [100*y_bins*i/y_intensityC_sum for i in y_intensity]                   # Scale orig viewer so agrees with binned
            y_vd = [ p for p,n in zip(y_pos, y_intensityC) for _ in range(n) ]                    # list of positions to be histogrammed

            # Creating ecat histograms if they are used
            if all_x is not None:
                # Find position index using viewer
                posV = viewers[viewer]["index"]
                posS = element_names_stripped.index('FP1 Slits')  # position slits
                if posS < posV:   # Viewer is after fp1 slits
                    x = []
                    valid_indices = []
                    for i in range(len(all_x)):
                        if transmission_indices[i] >= posV \
                          and groupsInfo[group_number]['fp1_slits']["slits_mm"][0] <= all_x[i][posS]*1000 <= groupsInfo[group_number]['fp1_slits']["slits_mm"][1]:
                            valid_indices.append(i)
                            x.append( -all_x[i][posV]*1000 )
                    x = np.array(x)
                else:
                    x = np.array([-xx[posV]*1000 for i, xx in enumerate(all_x) if transmission_indices[i] >= posV]) # in mm
                # Scale ecat data to viewer counts
                x_edges = np.histogram_bin_edges(x_vd, bins=x_bins)
                vd_counts, _ = np.histogram(x_vd, bins=x_edges)
                ecat_counts, _ = np.histogram(x, bins=x_edges)
                widths = np.diff(x_edges)
                area_vd   = np.sum(vd_counts * widths)
                area_ecat = np.sum(ecat_counts * widths)
                x_scale = area_vd / area_ecat if area_ecat > 0 else 0.0   # weights of ECAT x-histogram
                # Calculating chi-square
                viewer_counts, edges = np.histogram(x_vd, bins=x_bins)
                ecat_counts, edges = np.histogram(x, bins=x_edges, weights=np.full(len(x), x_scale))
                chisq_x = np.sum( [ (viewer_counts[i]-ecat_counts[i])**2 for i in range(len(viewer_counts)) ] )/np.sum(viewer_counts)

            if all_y is not None and all_x is not None:
                # Find position index using viewer
                if element_names_stripped.index('FP1 Slits') < posV:   # Viewer is after fp1 slits
                    y = np.array([-yy[posV]*1000 for i, yy in enumerate(all_y) if i in valid_indices]) # in mm
                else:
                    y = np.array([-yy[posV]*1000 for i, yy in enumerate(all_y) if transmission_indices[i] >= posV]) # in mm
                # Scale ecat data to viewer counts
                y_edges = np.histogram_bin_edges(y_vd, bins=y_bins)
                vd_counts, _ = np.histogram(y_vd, bins=y_edges)
                ecat_counts, _ = np.histogram(y, bins=y_edges)
                widths = np.diff(y_edges)
                area_vd   = np.sum(vd_counts * widths)
                area_ecat = np.sum(ecat_counts * widths)
                y_scale = area_vd / area_ecat if area_ecat > 0 else 0.0  # weights of ECAT y-histogram
                # Calculating chi-square
                viewer_counts, edges = np.histogram(y_vd, bins=y_bins)
                ecat_counts, edges = np.histogram(y, bins=y_edges, weights=np.full(len(y), y_scale))
                chisq_y = np.sum( [ (viewer_counts[i]-ecat_counts[i])**2 for i in range(len(viewer_counts)) ] )/np.sum(viewer_counts)

        if metric == 'chisq_2d' or plot == True:
            image_path = os.path.join(group_path, f"Group{group_number}_VD{viewer}_2D_cosy.npz")
            # Load 2d viewer data
            data = np.load(image_path)
            corrected_full_image = data["corrected_full_image"]
            corrected_full_image = corrected_full_image[:, ::-1]
            x2d_range = data["xf_range"]
            y2d_range = data["yf_range"]

    	# Optionally plot the projection
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(17,9))
            labelPP    = groupsInfo[group_number]['pp_alignment']['label']
            labelSlits = groupsInfo[group_number]['fp1_slits']['label']
            plt.suptitle(f"Group {group_number} - {labelPP} - {labelSlits} - {viewer}\n(Viewing with the beam)", fontsize=16)

            axs[0,0].plot(x_pos, x_intensity, label = 'Viewer original')
            axs[0,0].hist(x_vd, bins=x_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
            if all_x is not None:
                axs[0,0].hist(x, bins=x_edges, histtype='step', linewidth=4, weights=np.full(len(x), x_scale),  # uniformly scale ecat
                    label=f'ecat × {x_scale:.3g}', color = 'orange')
            axs[0,0].set_xlabel("X [mm]")
            axs[0,0].set_ylabel("Intensity")
            axs[0,0].set_title("X projection", fontweight = 'bold')
            axs[0,0].set_xlim(min(x_pos), max(x_pos))
            axs[0,0].legend(loc='best')   # place legend
            axs[0,0].text( 0.01, 0.98, f"chi-sq = {chisq_x:.1f}", transform=axs[0,0].transAxes, ha="left", va="top", fontsize=12, color = 'orange', fontweight='bold')

            axs[0,1].plot(y_pos, y_intensity, label = 'Viewer original')
            axs[0,1].hist(y_vd, bins=y_bins, histtype='step', color='blue', linewidth=4, label = 'Viewer binned')
            if all_y is not None:
                axs[0,1].hist(y, bins=y_edges, histtype='step', linewidth=4, weights=np.full(len(y), y_scale),  # uniformly scale ecat
                    label=f'ecat × {y_scale:.3g}', color = 'orange')
            axs[0,1].set_xlabel("Y [mm]")
            axs[0,1].set_ylabel("Intensity")
            axs[0,1].set_title("Y projection", fontweight = 'bold')
            axs[0,1].set_xlim(min(y_pos), max(y_pos))
            axs[0,1].legend(loc='best')   # place legend
            axs[0,1].text( 0.01, 0.98, f"chi-sq = {chisq_y:.1f}", transform=axs[0,1].transAxes, ha="left", va="top", fontsize=12, color = 'orange', fontweight='bold')

            im = axs[1,0].imshow( corrected_full_image, extent = (-x2d_range[-1], -x2d_range[0], y2d_range[-1], y2d_range[0]), \
                cmap = 'viridis', aspect = 'auto')
            cbar = fig.colorbar(im, ax=axs[1,0])
            axs[1,0].set_xlim(-x2d_range[-1], -x2d_range[0])
            axs[1,0].set_ylim(y2d_range[0], y2d_range[-1])
            axs[1,0].set_xlabel('X (mm)')
            axs[1,0].set_ylabel('Y (mm)')
            axs[1,0].set_title("Viewer", fontweight = 'bold')

            if all_x is not None and all_y is not None:
                axs[1,1].hist2d(x, y, bins=(50, 50), cmap='viridis',range=[[-x2d_range[-1], -x2d_range[0]],[y2d_range[0], y2d_range[-1]]])
                axs[1,1].set_xlabel('X (mm)')
                axs[1,1].set_ylabel('Y (mm)')
                axs[1,1].set_title("ECAT", fontweight = 'bold')

            plt.tight_layout()

            if save_fig:
                save_path = f"{results_directory}/imageCompared_Group{group_number}_VD{viewer}"
                plt.savefig(save_path, dpi = 300)
            else:
                plt.show()
        if metric == 'chisq_1d':
            metric_value = metric_value + chisq_x + chisq_y

    return metric_value

# Function that counts the number of SECAR parameters to be optimized and creates an array of bounds
def countSECARparams(params):
    count = 0
    pbounds = []
    for p in params:
        if p['par'] == 'XY':
            pbounds.append(bounds['dXY'])
            pbounds.append(bounds['dXY'])
            count += 2
        else:
            pbounds.append(bounds[p['par']])
            count += 1
    return count, pbounds

# Creates labels for corner plot
def make_param_labels(params):
    labels = []
    for p in params:
        if p['par'] == "XY":
            labels.append(f"{p['elem']}_X")
            labels.append(f"{p['elem']}_Y")
        else:
            labels.append(f"{p['elem']}_{p['par']}")
    return labels

# Function to convert emcee input to something runECAT can use
def convert_theta_to_values(params, theta):
    """Map a flat theta vector onto params structure, expanding XY -> [x,y]."""
    values, j = [], 0
    for p in params:
        if p["par"] == "XY":
            values.append([theta[j], theta[j+1]])
            j += 2
        else:
            values.append(theta[j])
            j += 1
    return values

# Function to create arrays tuneChanges and initialDistChanges (parameters in runECAT)
def prepareMCChanges(params, values):
    tuneChanges = []
    initialDistChanges = []
    flagCosy = False
    for i, p in enumerate(params):
        if p['elem'] != 'initialDist':
            tuneChanges.append( [ p['elem'], p['par'], values[i] ] )
            if p['par'] != 'dB':
                flagCosy = True
        else:
            initialDistChanges.append( [ p['par'], values[i] ] )
    return tuneChanges, initialDistChanges, flagCosy

# Prior to penalize large changes
def log_prior(theta, pbounds):
    i = 0
    for i, t in enumerate(theta):
        if t < pbounds[i][0] or t > pbounds[i][1] : # penalty if out of bounds
                return -np.inf
    return 0

def log_posterior(theta, params, pbounds, images2compare):
    if log_prior(theta, pbounds) == 0:
        try:
            values = convert_theta_to_values(params, theta)
            tuneChanges, initialDistChanges, run_cosy_flag = prepareMCChanges(params, values)
            metric = 0
            for groupImages in images2compare:
                group = groupImages[0]
                imagesByGroup = groupImages[1]
                initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
                                        'nRays': 5,
                                        'target_alignment' :   {"X": groupsInfo[group]['target_alignment']['X'], "Y": groupsInfo[group]['target_alignment']['Y'], "R": groupsInfo[group]['target_alignment']['R']},
                                        'aperture_alignment' : {"X": groupsInfo[group]['pp_alignment']['X'], "Y": groupsInfo[group]['pp_alignment']['Y'], "R": groupsInfo[group]['pp_alignment']['R'], 'separation_distance': 0.48619},
                                        'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0/1000, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
                                        'dE' : {'option': 'normal', 'param' : {'mu':  2/100, 'sigma': 8/100}},      # 2, 8
                                        'dZ' : {'option': 'fixed',  'param': 0}, }
                if len(initialDistChanges) != 0:  # Make changes to initialDistribution if needed
                    initialDistribution['angles']['mu_y'] = initialDistChanges[0][1]
                all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes = \
                    runECAT( initialDistribution = initialDistribution, run_cosy_flag = run_cosy_flag, tuneChanges = tuneChanges,
                        save_rays_flag = False, final_index = viewers[f"{np.max( [int(v) for v in imagesByGroup] )}"]["index"] )

                metric = metric + load_projection(imagesByGroup, group, all_x = all_x, all_y = all_y, transmission_indices = transmission_indices,
                    plot = False, metric = 'chisq_1d')
            return -metric
        except Exception:
            return -np.inf
    else:
        return -np.inf

# class created so multiprocessing can work more efficiently: restart processes
# that hang and after a certain number of tasks (prevents memory leaks)
class TimedRestartingPool:
    """
    Drop-in replacement for emcee's pool=... argument.
    - Provides map(func, iterable) with a per-task timeout.
    - Uses multiprocessing.Pool with maxtasksperchild to restart leaky workers.
    """
    def __init__(self, processes=None, timeout=120, maxtasksperchild=200, start_method="spawn"):
        self.timeout = timeout
        self._ctx = mp.get_context(start_method)
        self._pool = self._ctx.Pool(processes=processes, maxtasksperchild=maxtasksperchild)

    def map(self, func, iterable):
        # one task per item so timeout applies individually
        asyncs = [self._pool.apply_async(func, (x,)) for x in iterable]
        out = []
        for a in asyncs:
            try:
                out.append(a.get(timeout=self.timeout))
            except mp.TimeoutError:
                # Treat timeouts as invalid points so the sampler can keep going
                out.append(-np.inf)
            except Exception:
                # You can choose to re-raise; returning -inf keeps the run alive
                out.append(-np.inf)
        return out

    def close(self):
        self._pool.close()
        self._pool.join()

    # context-manager support
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        self.close()

# Run MCMC optimization
def optimizeMCMC(params, images2compare, fresh_start = True):
    '''
    params: list of parameters to optimize
    fresh_start: start from scratch or from previous optimization True/False
    '''
    rng = np.random.default_rng(42)

    # --- Run MCMC ---
    ndim, pbounds = countSECARparams(params)
    nwalkers, nsteps = 12, 20

    if fresh_start:
        os.system(f"rm -f chain.h5")
        backend = HDFBackend("chain.h5")    # Save as you go in case we want to restart optimization
        backend.reset(nwalkers, ndim)
        p0 = np.column_stack([rng.uniform(pbounds[i][0], pbounds[i][1], nwalkers) for i in range(ndim)])
    else:
        backend = HDFBackend("chain.h5")    # Save as you go in case we want to restart optimization
        p0 = backend.get_last_sample()

    logp = partial(log_posterior, params=params, pbounds = pbounds, images2compare = images2compare)
    nprocs = max(1, mp.cpu_count() - 1)

    # Per-eval timeout (seconds) and periodic worker restarts
    EVAL_TIMEOUT = 120          # ~2–3× typical runECAT time
    MAXTASKS_PER_CHILD = 200    # restart workers every ~200 tasks
    with TimedRestartingPool(processes=nprocs,timeout=EVAL_TIMEOUT, maxtasksperchild=MAXTASKS_PER_CHILD, start_method="spawn") as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logp, backend=backend, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    # --- Extract samples (discard burn-in, thin a bit) ---
    burn = nsteps // 2
    thin = 1
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)          # (Nsamples, ndim)
    logp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)       # (Nsamples,)
    if flat.size == 0:
        raise RuntimeError("No samples left after burn/thin. Increase nsteps or reduce burn/thin.")

    # Posterior medians and 16/84% intervals (per parameter)
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)

    # Highest posterior probability among draws
    idx_opt = int(np.argmax(logp))
    theta_opt = flat[idx_opt].copy()
    print("\n--- Posterior summaries ---")
    labels = make_param_labels(params)         # one label per dimension in theta
    for k, SECARpar in enumerate( labels ):
        print(f"{SECARpar:12s}: median={q50[k]: .6g}  [16%,84%]=[{q16[k]: .6g}, {q84[k]: .6g}]")
    print(f"\nHighest posterior probability among draws: {theta_opt} with a metric = {-np.max(logp):.1f}\n")

    # Show best fit result
    values_opt = convert_theta_to_values(params, theta_opt)
    tuneChanges, initialDistChanges, run_cosy_flag = prepareMCChanges(params, values_opt)
    metric = 0
    for groupImages in images2compare:
        group = groupImages[0]
        imagesByGroup = groupImages[1]
        initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
                                'nRays': 5,
                                'target_alignment' :   {"X": groupsInfo[group]['target_alignment']['X'], "Y": groupsInfo[group]['target_alignment']['Y'], "R": groupsInfo[group]['target_alignment']['R']},
                                'aperture_alignment' : {"X": groupsInfo[group]['pp_alignment']['X'], "Y": groupsInfo[group]['pp_alignment']['Y'], "R": groupsInfo[group]['pp_alignment']['R'], 'separation_distance': 0.48619},
                                'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0/1000, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
                                'dE' : {'option': 'normal', 'param' : {'mu':  2/100, 'sigma': 8/100}},      # 2, 8
                                'dZ' : {'option': 'fixed',  'param': 0}, }
        if len(initialDistChanges) != 0:  # Make changes to initialDistribution if needed
            initialDistribution['angles']['mu_y'] = initialDistChanges[0][1]
        all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes = \
            runECAT( initialDistribution = initialDistribution, run_cosy_flag = run_cosy_flag, tuneChanges = tuneChanges,
                save_rays_flag = False, final_index = viewers[f"{np.max( [int(v) for v in imagesByGroup] )}"]["index"] )

        metric = metric + load_projection(imagesByGroup, group, all_x = all_x, all_y = all_y, transmission_indices = transmission_indices,
            plot = False, metric = 'chisq_1d')
    print(f"\nHighest posterior probability among draws (again): {values_opt} with a metric = {metric:.1f}\n")

    # Create corner plot
    fig = corner.corner(flat, labels=labels, quantiles=[0.16, 0.50, 0.84], show_titles=True,
        title_fmt=".3g", truths=theta_opt )
    fig.savefig(f"{results_directory}/posterior_corner.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
