import os, re
import numpy as np

###################################
##### DEFINE BASE DIRECTORIES #####
###################################

ECAT_directory = r'/Users/fernandomontes/Research/COSY/MC/ECAT'
COSY_directory = os.path.join(ECAT_directory, 'COSY10.2')
matrix_directory = os.path.join(ECAT_directory, 'Matrix_Files')
pkl_directory = os.path.join(ECAT_directory, 'pkl_Files')
results_directory = os.path.join(ECAT_directory, 'Results')

# Files for COSY and the resultant transportation matrices
base = "pg_June2025"
# base = "Shumpei_FP4"
fox_file = f"{base}"
matrix_file = f"{base}"
save_rays_file = f'{results_directory}/{base}_Rays.pkl'

# List of directories to create
dirs_to_create = [
    ECAT_directory,
    COSY_directory,
    matrix_directory,
    pkl_directory,
    results_directory
]

# Create each directory if it doesn't exist
for directory in dirs_to_create:
    os.makedirs(directory, exist_ok = True)

##################################
######### COSY line info #########
##################################
# Change if base cosy changes!!!!!!!!
# Dictionary with COSY .fox file line and nominal values information
cosy_lines = {
    'Q1': {
        'Roll': {'line': [296, 304], 'text1' : " RA ", 'nom': -0.0034, 'text2': "; {Roll}\n"},
         'dXY': {'line': [297, 303], 'text1' : " SA ", 'nom': [0.000051, 0.000145], 'text2': "; {x, y}\n"},
        'B_SC': {'line': 1010, 'text1' : "Q1_SC:= ", 'nom': 1.0, 'text2': "; \n"}
           },
    'Q2': {
        'Roll': {'line': [318, 326], 'text1' : " RA ", 'nom': 0.0073, 'text2': "; {Roll}\n"},
         'dXY': {'line': [319, 325], 'text1' : " SA ", 'nom': [-0.000295, 0.000223], 'text2': "; {x, y}\n"},
        'B_SC': {'line': 1011, 'text1' : "Q2_SC:= ", 'nom': 1.0, 'text2': "; \n"}
           },
    'B1': {
        'Roll': {'line': [347, 355], 'text1' : " RA ", 'nom': -0.0135, 'text2': "; {Roll}\n"},
         'dXY': {'line': [348, 354], 'text1' : " SA ", 'nom': [0.000404, 0.000261], 'text2': "; {x, y}\n"},
           },
    'B2': {
         'Roll': {'line': [366, 374], 'text1' : " RA ", 'nom': -0.0087, 'text2': "; {Roll}\n"},
          'dXY': {'line': [367, 373], 'text1' : " SA ", 'nom': [0.000434, 0.000279], 'text2': "; {x, y}\n"},
            },
    '1515': {
         'dist': {'line': [380, 383], 'text1' : "DL ", 'nom': [0.87087, 0.297381], 'text2': ";\n"},
            },
}

##################################
##### DEFINE SYSTEM GEOMETRY #####
##################################

# Define the distance a ray is propagated (by some matrix) before reaching the listed location
# NOTE: The number of elements and the corresonding distances MUST match those from the COSY .fox file(s)
elements_dict = {
    "Target Center (TA)": 0,
    "Q1 Entrance (TA-upQ1)": 0.800527,
    "Q1 Exit (upQ1-doQ1)": 0.2498,
    "D1477 Aperture Drive (doQ1-D1477)": 0.1060544,
    "Q2 Entrance (D1477-upQ2)": 0.0844356,
    "Q2 Exit (upQ2-doQ2)": 0.2979,
    # Need to check FC1485 position - These are the original positions
    "FC1485 (doQ2-FC1485)": 0.391283,
    "B1 Entrance (FC1485-upB1)": 0.189755,
    # Original Lengths: doQ2-FC1485 = 0.391283, FC1485-upB1 = 0.189755
    # Proposed Update: doQ2-FC1485 = 0.3926972, FC1485-upB1 = 0.1883408
    "B1 Exit (upB1-doB1)": 0.491123652,
    "B2 Entrance (doB1-upB2)": 0.999778,
    "B2 Exit (upB2-doB2)": 0.491137833,
    "VD1515 (doB2-VD1515)": 0.87087,
    "H1 Entrance (VD1515-upH1)": 0.297381,
    "H1 Exit (upH1-doH1)": 0.263,
    "Q3 Entrance (doH1-upQ3)": 0.268763,
    "Q3 Exit (upQ3-doQ3)": 0.3499,
    "Q4 Entrance (doQ3-upQ4)": 0.35139,
    "Q4 Exit (upQ4-doQ4)": 0.3467,
    "Q5 Entrance (doQ4-upQ5)": 0.213664,
    "Q5 Exit (upQ5-doQ5)": 0.3466,
    "FP1 Slits (doQ5-SLH1542)": 0.152895825,
    "VD1542 (SLH1542-VD1542)": 0.080755344,
    "B3 Entrance (VD1542-upB3)": 0.267380831,
    "B3 Exit (upB3-doB3)": 0.491574165,
    "B4 Entrance (doB3-upB4)": 0.509073,
    "B4 Exit (upB4-doB4)": 0.492634453,
    "H2 Entrance (doB4-upH2)": 0.297393,
    "H2 Exit (upH2-doH2)": 0.264,
    "AP1568 Aperture Drive (doH2-AP1568)": 0.23027568,
    "Q6 Entrance (AP1568-upQ6)": 0.30808432,
    "Q6 Exit (upQ6-doQ6)": 0.3395,
    "Q7 Entrance (doQ6-upQ7)": 0.196437,
    "Q7 Exit (upQ7-doQ7)": 0.3472,
    "WF1 Entrance (doQ7-upWF1)": 0.496188,
    "WF1 Exit (upWF1-doWF1)": 2.365,
    "H3 Entrance (doWF1-upH3)": 0.498516,
    "H3 Exit (upH3-doH3)": 0.263,
    "O1 Entrance (doH3-upO1)": 0.277156,
    "O1 Exit (upO1-doO1)": 0.262,
    "FP2 Slits (doO1-SLH1638)": 1.750011,
    "VD1638 (SLH1638-VD1638)": 0.28,
    "Q8 Entrance (VD1638-upQ8)": 0.59233,
    "Q8 Exit (upQ8-doQ8)": 0.2473,
    "Q9 Entrance (doQ8-upQ9)": 0.394688,
    "Q9 Exit (upQ9-doQ9)": 0.3037,
    "B5 Entrance (doQ9-upB5)": 0.358445,
    "B5 Exit (upB5-doB5)": 0.927206165,
    "B6 Entrance (doB5-upB6)": 0.35042,
    "B6 Exit (upB6-doB6)": 0.927206165,
    "VD1688 (doB6-VD1688)": 0.562676442,
    "Q10 Entrance (VD1688-upQ10)": 0.266293558,
    "Q10 Exit (upQ10-doQ10)": 0.2628,
    "Q11 Entrance (doQ10-upQ11)": 0.64758,
    "Q11 Exit (upQ11-doQ11)": 0.3424,
    "WF2 Entrance (doQ11-upWF2)": 0.997726,
    "WF2 Exit (upWF2-doWF2)": 2.365,
    "FP3 Slits & VD1783 (doWF2-SLH_VD1783)": 4.644225559,
    "Q12 Entrance (SLH_VD1783-upQ12)": 0.208449441,
    "Q12 Exit (upQ12-doQ12)": 0.2977,
    "Q13 Entrance (doQ12-upQ13)": 0.350425,
    "Q13 Exit (upQ13-doQ13)": 0.3008,
    "B7 Entrance (doQ13-upB7)": 0.659564,
    "B7 Exit (upB7-doB7)": 1.199913861,
    "B8 Entrance (doB7-upB8)": 0.680262,
    "B8 Exit (upB8-doB8)": 1.199913861,
    "VD1836 (doB8-VD1836)": 0.389887068,
    "Q14 Entrance (VD1836-upQ14)": 0.469972932,
    "Q14 Exit (upQ14-doQ14)": 0.2995,
    "Q15 Entrance (doQ14-upQ15)": 0.449858,
    "Q15 Exit (upQ15-doQ15)": 0.3012,
    "UMCP (doQ15-UMCP)": 0.7112,
    "DMCP (UMCP-DMCP)": 1.397508,
    "FP4 Slits (DMCP-SLH1879)": 0.512209509,
    "VD1879 (SLH1879-VD1879)": 0.208469264,
    "DSSD (VD1879-DSSD)": 0.3302
}

# Path to the pkl file containing the system's full beamline geometry used to check transmission
# This is built using the "build_beamline.py" script (stored in the "Beampipes" folder)
beamline_geometry = f'{pkl_directory}/SECAR_Beamline.pkl'

#################################
##### DERIVED BEAMLINE AXIS #####
#################################

# Extract all the keys and, for plotting purposes, the stripped keys only listing the ray positions at evaluation
# (i.e., removing the matrix names from the keys)
element_names_full = list(elements_dict.keys())
element_names_stripped = [re.sub(r'\s*\(.*?\)', '', name) for name in element_names_full]

# Extract the propagation distances (in meters) for each beamline segment and compute the cumulative z-positions
beam_axis = [elements_dict[key] for key in elements_dict]
z_coordinates = [sum(beam_axis[:i+1]) for i in range(len(beam_axis))]

#################################
### DIAGNOSTICS and DETECTORS ###
#################################

viewers = {
    '1515': {
        'size_mm': (6.93*25.4, 4.33*25.4/np.sqrt(2)),
        'misalignment': (-1.65, 2.5),
        'index': element_names_stripped.index('VD1515')
    },
    '1542': {
        'size_mm': (1.77*25.4, 2.5*25.4/np.sqrt(2)),
        'misalignment': (-6.17, 0.37),
        'index': element_names_stripped.index('VD1542')
    },
    '1638': {
        'size_mm': (5.0*25.4, 3.346*25.4/np.sqrt(2)),
        'misalignment': (-5.05, -0.24),
        'index': element_names_stripped.index('VD1638')
    },
    '1688': {
        'size_mm': (3.15*25.4/np.sqrt(2), 4.45*25.4),
        'misalignment': (0.43, -0.63),
        'index': element_names_stripped.index('VD1688')
    },
    '1783': {
        'size_mm': (5.28*25.4, 4.75*25.4/np.sqrt(2)),
        'misalignment': (0.45, -0.3),
        'index': element_names_stripped.index('FP3 Slits & VD1783')
    },
    '1836': {
        'size_mm': (4.45*25.4, 3.15*25.4/np.sqrt(2)),
        'misalignment': (2.58, -0.9),
        'index': element_names_stripped.index('VD1836')
    },
    '1879': {
        'size_mm': (4.25*25.4/np.sqrt(2), 2.25*25.4),
        'misalignment': (-1.59, 0.82),
        'index': element_names_stripped.index('VD1879')
    }
}

apertures = {
    'D1477':    {'index': element_names_stripped.index('D1477 Aperture Drive'),
                 'misalignment': (0, 0)},
    'AP1568':   {'index': element_names_stripped.index('AP1568 Aperture Drive'),
                 'misalignment': (0, 0)},
}

detectors = {
    'UMCP': {'index': element_names_stripped.index('UMCP'),
             'misalignment': (0, 0)},
    'DMCP': {'index': element_names_stripped.index('DMCP'),
             'misalignment': (0, 0)},
    'DSSD': {'size_mm': (64, 64),
             'index': element_names_stripped.index('DSSD'),
             'misalignment': (0, 0)},
}

faraday_cups = {
    '1485': {'size_mm': (3.752*25.4, 0.978*25.4),
               'index': element_names_stripped.index('FC1485'),
               'misalignment': (0, 0)},
    '1542': {'size_mm': (1.661*25.4, 1.633*25.4),
               'index': element_names_stripped.index('VD1542'),
               'misalignment': (-6.17, 0.37)},
    '1638': {'size_mm': (4.661*25.4, 2.091*25.4),
               'index': element_names_stripped.index('VD1638'),
               'misalignment': (-5.05, -0.24)},
    '1798': {'size_mm': (1.339*25.4, 1.181*25.4),
               'index': element_names_stripped.index('Q13 Exit'),   # NEED TO CHECK THIS!!!!!!!!!
               'misalignment': (0, 0)},
    '1879': {'size_mm': (3.11*25.4, 2.32*25.4),
               'index': element_names_stripped.index('VD1879'),
               'misalignment': (-1.59, 0.82)},
}
