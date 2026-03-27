import numpy as np
import matplotlib.pyplot as plt
from emcee.backends import HDFBackend
import corner

from helper_functions import runECAT
from config import viewers, element_names_stripped, results_directory, fox_file

from ViewerAnalysis import convert_theta_to_values, groupsInfo, prepareMCChanges, load_projection, load_just_projection

# Creates labels for corner plot and creates multipliers to convert from meters to mm, fraction to percentage and rad to mrad
def make_param_labels(params):
    labels = []
    paramScale = []
    for p in params:
        if p['par'] == "XY":
            labels.append(f"{p['elem']} X [mm]")
            labels.append(f"{p['elem']} Y [mm]")
            paramScale.append(1000)  # meters to mm
            paramScale.append(1000)  # meters to mm
        elif p['par'] == "X" or p['par'] == "Y":
            labels.append(f"{p['elem']} {p['par']} [mm]")
            paramScale.append(1000)  # meters to mm
        elif p['par'] == "dB" or p['par'] == "B_SC":
            name = p['elem'][:3]
            labels.append(f"{name} mistuned [%]")
            paramScale.append(100)  # fraction to percentage
        elif p['par'] == "mu_y":
            labels.append(f"{p['elem']} Angle Y [mrad]")
            paramScale.append(1000)  # rad to mrad
        elif p['par'] == "mu" or p['par'] == "sigma":
            labels.append(f"{p['elem']} Energy {p['par']} [%]")
            paramScale.append(100)  # fraction to percentage
        elif p['par'] == "Roll":
            labels.append(f"{p['elem']} Roll [mrad]")
            paramScale.append(1000)  # rad to mrad
    return labels, paramScale

# Plot MCMC optimization
def PlotMCMCresults(params, images2compare = None, plotParameters = True):
    '''
    params: list of parameters that have been used in the optimization
    '''

    # --- Load saved MCMC  ---
    backend = HDFBackend("chain.h5", read_only=True)    # Read saved optimization
    iters = backend.iteration
    if iters == 0:
        raise RuntimeError("chain.h5 contains no samples yet.")

    # --- Extract samples (discard burn-in, thin a bit) ---
    burn = iters // 2
    thin = 1
    # Get flattened samples and log-probabilities
    flat = backend.get_chain(discard=burn, thin=thin, flat=True)          # (Nsamples, ndim)
    logp = backend.get_log_prob(discard=burn, thin=thin, flat=True)       # (Nsamples,)
    flat = backend.get_chain(flat=True)          # (Nsamples, ndim)
    logp = backend.get_log_prob(flat=True)       # (Nsamples,)

    if flat.size == 0:
        raise RuntimeError("No samples left after burn/thin. Increase nsteps or reduce burn/thin.")
    # Highest posterior probability among draws
    idx_opt = int(np.argmax(logp))
    theta_opt_orig = flat[idx_opt].copy()
    labels, paramScale = make_param_labels(params)         # one label per dimension in theta
    paramScale = np.array(paramScale, dtype=float)
    flat = flat * paramScale          # column-wise scaling
    print(flat, logp)

    # Posterior medians and 16/84% intervals (per parameter)
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    theta_opt = flat[idx_opt].copy()    # in easy units [mm, mrad, %]
    print("\n--- Posterior summaries ---")
    for k, SECARpar in enumerate( labels ):
        print(f"{SECARpar:12s}: Param highest metric = {theta_opt[k]:.2f} , median = {q50[k]: .2f}  [16%,84%]=[{q16[k]: .2f}, {q84[k]: .2f}]")
    print(f"\nHighest posterior probability (metric) among draws: {-np.max(logp):.3f}\n")

    if plotParameters == True:
        mask = ~np.isinf(logp)
        print(len(logp))
        logp = logp[mask]
        print(len(logp), np.unique(logp).size)
        flat = flat[mask.ravel()]
        nuRow = 4
        nuCol = 4
        #np.savetxt("test.csv", np.column_stack([flat[:,0], logp]), delimiter=",", header="mu_y, logP", comments="")
        fig, axs = plt.subplots(nuRow, nuCol, figsize=(17,9))
        plt.suptitle("Chi-square vs parameter (MCMC)", fontsize=16)
        for j, SECARpar in enumerate( labels ):
            axs[int((j)/nuCol), (j)%nuCol].scatter(flat[:,j], logp, s=30,  alpha=0.8)
            axs[int((j)/nuCol), (j)%nuCol].set_xlabel(SECARpar)
            axs[int((j)/nuCol), (j)%nuCol].set_ylabel('chi-sq')
        plt.tight_layout()
        fig.savefig(f"{results_directory}/plotParameters.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    values_opt = convert_theta_to_values(params, theta_opt_orig)
    print(f"\nHighest posterior probability among draws (again): {values_opt}\n")

    if images2compare != None:
        # Show best fit result
        tuneChanges, initialDistChanges, run_cosy_flag = prepareMCChanges(params, values_opt)
        # tuneChangesPrev = [['Q1', 'Y', -0.0004005534288656388], ['Q1', 'B_SC', 0.01016527258784548], ['Q2', 'Y', -0.0005769743274796191], ['Q2', 'B_SC', -0.006159636853166115 ], ['Q3', 'B_SC', 0.00036907811424478215], ['Q4', 'B_SC', -0.013252101342486288], ['Q5', 'B_SC', 0.006975071308727706], ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['Q6', 'B_SC', -0.026865801259426076], ['Q7', 'B_SC', -0.006350239818653578], ['WF1 Exit', 'dB', -0.0125], ['B5 Exit', 'dB', -0.0053], ['B6 Exit', 'dB', -0.0053], ['WF2 Exit', 'dB', -0.0059]]
        # tuneChanges = tuneChangesPrev + tuneChanges

        tuneChangesPrev = [ ['B3 Exit', 'dB', +0.01], ['B4 Exit', 'dB', +0.01], ['WF1 Exit', 'dB', -0.0125], ['B5 Exit', 'dB', -0.0053], ['B6 Exit', 'dB', -0.0053], ['WF2 Exit', 'dB', -0.0059]]
        tuneChanges = tuneChangesPrev
        run_cosy_flag = True

        # tuneChanges = None      # uncomment this to run nominal optics
        # run_cosy_flag = False   # uncomment this to run nominal optics
        metric = 0
        cosyFile = fox_file
        for groupImages in images2compare:
            group = groupImages[0]
            imagesByGroup = groupImages[1]
            initialDistribution = { 'type': 'aperture',   # Generates nRays originating from a circular target and determines which are transmitted through a circular downstream aperture.
                                    'nRays': 7000,
                                    'target_alignment' :   {"X": groupsInfo[group]['target_alignment']['X'], "Y": groupsInfo[group]['target_alignment']['Y'], "R": groupsInfo[group]['target_alignment']['R']},
                                    'aperture_alignment' : {"X": groupsInfo[group]['pp_alignment']['X'], "Y": groupsInfo[group]['pp_alignment']['Y'], "R": groupsInfo[group]['pp_alignment']['R'], 'separation_distance': 0.48619},
                                    'angles' : {'mu_x' : -0.12/1000, 'mu_y' : 0.0043393, 'sigma' : 10.24/1000},    # mu_y = 15 mrad
                                    'dE' : {'option': 'normal', 'param' : {'mu':  0.020249, 'sigma': 0.077522}},      # 2, 8
                                    'dZ' : {'option': 'fixed',  'param': 0}, }
            for initDist in initialDistChanges:  # Make changes to initialDistribution if needed
                if initDist[0] == 'mu_y':
                    initialDistribution['angles']['mu_y'] = initDist[1]
                elif initDist[0] == 'mu':
                    initialDistribution['dE']['param']['mu'] = initDist[1]
                elif initDist[0] == 'sigma':
                    initialDistribution['dE']['param']['sigma'] = initDist[1]
            all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, chamber_names, beampipes, cosyFile = \
                runECAT( initialDistribution = initialDistribution, run_cosy_flag = run_cosy_flag, foxFile = cosyFile, tuneChanges = tuneChanges,
                    save_rays_flag = False, final_index = viewers[f"{max(int(n) for _, nums in images2compare for n in nums)}"]["index"] )
            run_cosy_flag = False # Only needs to run cosy once for every MCMC iteration
            metric = metric + load_projection(imagesByGroup, group, all_x = all_x, all_y = all_y, transmission_indices = transmission_indices,
                plot = True, metric = 'chisq_1d', save_fig = True)
        print(metric/160000.0)
        print(f"\nHighest posterior probability among draws (again): {values_opt} with a metric = {metric:.1f}\n")

    # Create corner plot
    #fig = corner.corner(flat, labels=labels, quantiles=[0.16, 0.50, 0.84], show_titles=True,
    #    title_fmt=".3g", truths=theta_opt )
    fig = corner.corner(flat, labels=labels, levels = (0.393,), show_titles=True,
        title_fmt=".3g", truths=theta_opt )
    fig.savefig(f"{results_directory}/posterior_corner.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# params = [
#     # {'elem': 'initialDist', 'par': 'mu_y'},
#     # {'elem': 'initialDist', 'par': 'mu'},
#     # {'elem': 'initialDist', 'par': 'sigma'},
#     # {'elem': 'Q1',          'par': 'Y'},
#     # {'elem': 'Q1',          'par': 'B_SC'},
#     # {'elem': 'Q2',          'par': 'Y'},
#     # {'elem': 'Q2',          'par': 'B_SC'},
#     # {'elem': '1515',        'par': 'dist'},
#     # {'elem': 'B1',          'par': 'Y'},
#     # {'elem': 'B1 Exit',     'par': 'dB'},
#     # {'elem': 'B2',          'par': 'Y'},
#     # {'elem': 'B2 Exit',     'par': 'dB'},
#     # {'elem': 'Q3',          'par': 'B_SC'},
#     # {'elem': 'Q4',          'par': 'B_SC'},
#     # {'elem': 'Q5',          'par': 'B_SC'},
#     # {'elem': 'B3 Exit',     'par': 'dB'},
#     # {'elem': 'B4 Exit',     'par': 'dB'},
#     # {'elem': 'Q6',          'par': 'B_SC'},
#     # {'elem': 'Q7',          'par': 'B_SC'},
#     # {'elem': 'WF1 Exit',    'par': 'dB'},
#     {'elem': 'Q8',          'par': 'B_SC'},
#     {'elem': 'Q9',          'par': 'B_SC'},
#     {'elem': 'Q9',          'par': 'X'},
#     {'elem': 'Q10',          'par': 'B_SC'},
#     {'elem': 'Q11',          'par': 'B_SC'},
#     {'elem': 'WF2 Exit',    'par': 'dB'},
# ]
#images2compare = [ ["7", ["1515", "1542"]], ["8", ["1515", "1542"]], ["9", ["1542"]], ["10", ["1515", "1542"]], ["14", ["1515", "1542"]], ["15", ["1515", "1542"]], ["16", ["1515"]] ] #, ["17", ["1515"]] ]
#images2compare = [ ["7", ["1638"]], ["8", ["1638"]], ["9", ["1638"]], ["10", ["1638"]], ["13", ["1638"]], ["14", ["1638"]], ["15", ["1638"]] ]
#images2compare = [ ["7", ["1515", "1542", "1638"]], ["8", ["1515", "1542", "1638"]], ["9", ["1542", "1638"]], ["10", ["1515", "1542", "1638"]], ["13", ["1638"]], ["14", ["1515", "1542", "1638"]], ["15", ["1515", "1542", "1638"]], ["16", ["1515"]] ]
#images2compare = [ ["7", ["1638"]], ["8", ["1638"]], ["9", ["1638"]], ["10", ["1638"]], ["13", ["1638"]], ["14", ["1638"]], ["15", ["1638"]] ]
#images2compare = [ ["7", ["1638", "1688", "1783"]], ["8", ["1638", "1688", "1783"]], ["9", ["1638", "1688", "1783"]], ["10", ["1638", "1783"]], ["13", ["1638"]], ["14", ["1638", "1688", "1783"]], ["15", ["1638", "1688", "1783"]] ]
#images2compare = [ ["7", ["1688", "1783"]], ["8", ["1688", "1783"]],  ["9", ["1688", "1783"]], ["10", ["1783"]], ["14", ["1688", "1783"]],  ["15", ["1688", "1783"]] ]
#images2compare = [ ["7", ["1688", "1783"]],  ["10", ["1783"]] ]
#images2compare = [  ["10", ["1783"]] ]
#images2compare = [ ["7", ["1836", "1879"]], ["8", ["1836", "1879"]], ["9", ["1836", "1879"]], ["11", ["1879"]], ["13", ["1879"]],  ["15", ["1836", "1879"]]]

images2compare = [ ["7", ["1515", "1542", "1638", "1688", "1783"]], ["8", ["1515", "1542", "1638", "1688", "1783"]],  ["9", ["1542", "1638","1688", "1783"]], ["10", ["1515", "1542", "1638", "1783"]], ["13", ["1638"]], ["14", ["1515", "1542", "1638", "1688", "1783"]],  ["15", ["1515", "1542", "1638", "1688", "1783"]], ["16", ["1515"]] ]

#PlotMCMCresults(params, images2compare = images2compare)
#load_just_projection(images2compare)
