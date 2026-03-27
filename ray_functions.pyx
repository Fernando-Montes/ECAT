# To get cython running
# python setup.py build_ext --inplace

# cython: language_level = 3
# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport sin, cos, sqrt, acos, atan2, M_PI

import numpy as np
cimport numpy as np
ctypedef np.float64_t DTYPE_t
cdef Py_ssize_t i, n

from typing import List
from scipy.stats import skewnorm
from shapely import contains_xy
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

###################################
##### GENERATING INITIAL RAYS #####
###################################

# Function to generate stepped values within a range
def generate_stepped_values(min, max, num_steps):

    if num_steps > 1:
        step_size = (max - min) / (num_steps - 1)

    elif num_steps == 1:
        step_size = (max - min) / num_steps

    return [min + i * step_size for i in range(num_steps)]

# Function to generate unfirom distributions with equal steps per coordinate for all coordinates (Mimics COSY's Method)
# The COSY condition mimics the ellipsoid phase space defined by (a/A)^2 + (b/B)^2 + (dE/E)^2 < 1
# used for generating the 189 characteristic SECAR rays
def stepped_uniform_rays(x_params, ax_params, y_params, ay_params, dE_params, dz_params, COSY_condition=False):

    # Generate the stepped values for each coordinate
    x_values = generate_stepped_values(x_params['min'], x_params['max'], x_params['steps'])
    ax_values = generate_stepped_values(ax_params['min'], ax_params['max'], ax_params['steps'])
    y_values = generate_stepped_values(y_params['min'], y_params['max'], y_params['steps'])
    ay_values = generate_stepped_values(ay_params['min'], ay_params['max'], ay_params['steps'])
    dE_values = generate_stepped_values(dE_params['min'], dE_params['max'], dE_params['steps'])
    dz_values = generate_stepped_values(dz_params['min'], dz_params['max'], dz_params['steps'])

    # Set both l and dm to 0
    cdef double l = 0.0
    cdef double dm = 0.0

    # Generate all possible combinations
    combos = np.array(np.meshgrid(x_values, ax_values, y_values, ay_values, dE_values, dz_values)).T.reshape(-1, 6)

    # Store list of all rays
    rays = []

    # Extract maximums for COSY_condition check
    ax_max = max(abs(ax_params['min']), abs(ax_params['max']))
    ay_max = max(abs(ay_params['min']), abs(ay_params['max']))
    dE_max = max(abs(dE_params['min']), abs(dE_params['max']))

    for combo in combos:
        x_value, ax_value, y_value, ay_value, dE_value, dz_value = combo

        # COSY acceptance ellipse check
        if COSY_condition:
            cosy_metric = (
                (ax_value / ax_max)**2 +
                (ay_value / ay_max)**2 +
                (dE_value / dE_max)**2
            )
            if cosy_metric > 1.01:
                continue

        # Construct the ray
        ray = (
            [x_value],
            [ax_value],
            [y_value],
            [ay_value],
            [l],
            [dE_value],
            [dm],
            [dz_value]
        )
        rays.append(ray)

    return rays

# To assist with generating values from common distributions
cdef double sample_dist(str option, dict params, int seed):
    np.random.seed(seed)
    if option == 'fixed':
        return <double>params['fixed']
    elif option == 'normal':
        return np.random.normal(params['mu'], params['sigma'])
    elif option == 'uniform':
        return np.random.uniform(params['min'], params['max'])
    elif option == 'stepped':
        return np.random.choice( np.linspace(params['min'], params['max'], params['steps']) )
    elif option == 'skewed':
        return skewnorm.rvs(params['alpha'], loc=params['mu'], scale=params['sigma'])
    else:
        raise ValueError(f"Unknown option: {option}")

# Function to generate rays transmitted through a downstream aperture
def aperture_rays(target_radius, target_offset_x, target_offset_y,
              aperture_radius, aperture_offset_x, aperture_offset_y,
              separation_distance,
              theta_mu_x, theta_mu_y, theta_sigma,
              dE_option, dE_params,
              dz_option, dz_params, nRays,
              return_positions = False, return_angles = False, max_factor = 1000):

    cdef double x, y, ax, ay, dE, dz
    cdef double l = 0.0
    cdef double dm = 0.0

    np.random.seed(42)

    # Store list of all rays
    rays = []

    ##################################################
    ##### FOR DETERMING X, A (ax), Y, AND B (ay) #####
    ##################################################

    # Initialize empty lists for transmitted rays
    x_transmitted, y_transmitted, ax_transmitted, ay_transmitted, theta_transmitted, phi_transmitted = [], [], [], [], [], []

    # Initialize empty lists for position at aperture
    x_aperture, y_aperture = [], []

    # Start with a reasonable N and increase as needed
    initial_factor = 10

    # To ensure there are enough rays
    while len(x_transmitted) < nRays:

        # For debugging
        # print(len(x_transmitted), len(theta_transmitted), initial_factor)

        # Set N large to be able to have lists of transmission coordinates of greater length than nRays
        N = min(nRays * initial_factor, nRays * max_factor)

        # For debugging
        if initial_factor >= max_factor:
            raise MemoryError("Unable to generate enough rays within memory limits.")

        ##### GENERATE RAYS AT THE TARGET LOCATION, SAMPLING (X, Y) UNIFORMLY #####
        r_scalar = np.sqrt(np.random.uniform(0, 1, N))
        r_target = r_scalar * target_radius
        theta_target = np.random.uniform(0, 2 * np.pi, N)
        x_target = target_offset_x + (r_target * np.cos(theta_target))
        y_target = target_offset_y + (r_target * np.sin(theta_target))

        ##### SAMPLE THE ANGULAR DISTRIBUTION OUT OF THE TARGET (Assumes same σ) #####
        theta_x = np.random.normal(theta_mu_x, theta_sigma, N)
        theta_y = np.random.normal(theta_mu_y, theta_sigma, N)

        ##### DEFINE DIRECTION VECTORS & NORMALIZE TO GET A PROPER UNIT DIRECTION #####
        dx, dy, dz = theta_x, theta_y, 1.0
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        dir_x, dir_y, dir_z = dx / norm, dy / norm, dz / norm

        ##### PROPAGATE THE RAYS TO THE DOWNSTREAM APERTURE & CHECK TRANSMISSION #####
        scale = separation_distance / dir_z
        x_ap = x_target + (dir_x * scale)
        y_ap = y_target + (dir_y * scale)

        # Convert inputs to arrays if they are scalars
        aperture_offset_xs = np.atleast_1d(aperture_offset_x)
        aperture_offset_ys = np.atleast_1d(aperture_offset_y)
        aperture_radii     = np.atleast_1d(aperture_radius)

        # Sanity check
        if not (len(aperture_offset_xs) == len(aperture_offset_ys) == len(aperture_radii)):
            raise ValueError("Aperture parameter lengths must match")

        # Accepted if inside ANY aperture
        mask_any = np.zeros(N, dtype=bool)
        for ax, ay, ar in zip(aperture_offset_xs, aperture_offset_ys, aperture_radii):
            r_ap = np.sqrt((x_ap - ax)**2 + (y_ap - ay)**2)
            mask_any |= (r_ap < ar)
        accepted_rays_index = np.where(mask_any)[0]

        # r_ap = np.sqrt((x_ap - aperture_offset_x)**2 + (y_ap - aperture_offset_y)**2)
        # accepted_rays_index = np.where(r_ap < aperture_radius)[0]

        # Angular information of transmitted rays
        theta_vals = np.arccos(dir_z)
        phi_vals = np.arctan2(dir_y, dir_x)
        phi_deg_wrapped = (np.degrees(phi_vals) + 180) % 360 - 180  # Converts to degrees in [-180, 180]

        ##### STORE RESULTS #####

        # Positions at the aperture drive
        x_aperture.extend(x_ap[accepted_rays_index].tolist())
        y_aperture.extend(y_ap[accepted_rays_index].tolist())

        # Ray information at the target location
        x_transmitted.extend(x_target[accepted_rays_index].tolist())
        y_transmitted.extend(y_target[accepted_rays_index].tolist())

        # Angles for transmitted rays
        ax_transmitted.extend(theta_x[accepted_rays_index].tolist())
        ay_transmitted.extend(theta_y[accepted_rays_index].tolist())
        theta_transmitted.extend(theta_vals[accepted_rays_index].tolist())
        phi_transmitted.extend(phi_deg_wrapped[accepted_rays_index].tolist())

        # If we still don't have enough rays, increase N
        # initial_factor *= 2

    ###############################################################
    ##### SETTING INITIAL RAYS IN THE TARGET BASED ON RESULTS #####
    ###############################################################

    cdef list rays_list = []

    for i in range(nRays):

        x = x_transmitted[i]
        ax = ax_transmitted[i]
        y = y_transmitted[i]
        ay = ay_transmitted[i]
        dE = sample_dist(dE_option, dE_params, i)
        dz = sample_dist(dz_option, dz_params, i)

        """
        rays[i, 0] = x
        rays[i, 1] = ax
        rays[i, 2] = y
        rays[i, 3] = ay
        rays[i, 4] = l
        rays[i, 5] = dE
        rays[i, 6] = dm
        rays[i, 7] = dz
        """

        x = x_transmitted[i]
        ax = ax_transmitted[i]
        y = y_transmitted[i]
        ay = ay_transmitted[i]
        dE = sample_dist(dE_option, dE_params, i)
        dz = sample_dist(dz_option, dz_params, i)

        rays_list.append(([x], [ax], [y], [ay], [l], [dE], [dm], [dz]))

    # Return rays, and optionally theta and phi transmitted
    if return_angles and return_positions:
        return rays_list, x_transmitted, y_transmitted, x_aperture, y_aperture, theta_transmitted, phi_transmitted

    elif return_angles:
        return rays_list, theta_transmitted, phi_transmitted

    elif return_positions:
        return rays_list, x_transmitted, y_transmitted

    return rays_list

# Most general and flexible function for generating rays
# This is the original version where x, y, ax, and ay were handled independently with optional coupling
def mixed_rays_original(str x_option, dict x_params,
                        str ax_option, dict ax_params,
                        str y_option, dict y_params,
                        str ay_option, dict ay_params,
                        str dE_option, dict dE_params,
                        str dz_option, dict dz_params,
                        int nRays, bint circular_coupling = False):

    cdef np.ndarray[np.float64_t, ndim=2] rays = np.zeros((nRays, 8), dtype=np.float64)
    cdef int i
    cdef double x, y, ax, ay, dE, dz
    cdef double l = 0.0
    cdef double dm = 0.0
    cdef double r_scalar, theta
    cdef list rays_list = []

    for i in range(nRays):
        np.random.seed(i)

        # ----- SIMULATING COLLIMATOR POSITIONS ----- #
        if x_option == 'circle' and y_option == 'circle':
            r_scalar = sqrt(np.random.uniform(0.0, 1.0))
            theta = np.random.uniform(0.0, 2 * M_PI)
            x = x_params['center'] + r_scalar * x_params['radius'] * cos(theta)
            y = y_params['center'] + r_scalar * y_params['radius'] * sin(theta)

        else:
            # ----- X ----- #
            if x_option == 'circle':
                r_scalar = sqrt(np.random.uniform(0.0, 1.0))
                theta = np.random.uniform(0.0, 2 * M_PI)
                x = x_params['center'] + r_scalar * x_params['radius'] * cos(theta)
            else:
                x = sample_dist(x_option, x_params, i)

            # ----- Y ----- #
            if y_option == 'circle':
                r_scalar = sqrt(np.random.uniform(0.0, 1.0))
                theta = np.random.uniform(0.0, 2 * M_PI)
                y = y_params['center'] + r_scalar * y_params['radius'] * sin(theta)
            else:
                y = sample_dist(y_option, y_params, i)

        # ----- SIMULATING ISOTROPIC ANGLES ----- #
        if ax_option == 'circle' and ay_option == 'circle':
            r_scalar = sqrt(np.random.uniform(0.0, 1.0))
            theta = np.random.uniform(0.0, 2 * M_PI)
            ax = ax_params['center'] + r_scalar * ax_params['radius'] * cos(theta)
            ay = ay_params['center'] + r_scalar * ay_params['radius'] * sin(theta)

        elif ax_option == 'isotropic' and ay_option == 'isotropic':
            theta_max = ax_params.get('theta_max', 0.25)

            u = np.random.uniform(0.0, 1.0)
            theta_angle = acos(1.0 - u * (1.0 - cos(theta_max)))  # Isotropic in cone
            phi = np.random.uniform(0.0, 2.0 * M_PI)

            ax = theta_angle * cos(phi)
            ay = theta_angle * sin(phi)

        else:
            ax = sample_dist(ax_option, ax_params, i)
            ay = sample_dist(ay_option, ay_params, i)

        dE = sample_dist(dE_option, dE_params, i)
        dz = sample_dist(dz_option, dz_params, i)

        rays[i, 0] = x
        rays[i, 1] = ax
        rays[i, 2] = y
        rays[i, 3] = ay
        rays[i, 4] = l
        rays[i, 5] = dE
        rays[i, 6] = dm
        rays[i, 7] = dz

        rays_list.append(([x], [ax], [y], [ay], [l], [dE], [dm], [dz]))

    return rays_list

# Updated mixed_rays function which makes it easier to handle cases where positions (x, y) and angles (ax, ay) are combined
def mixed_rays(str position_option, dict position_params,
               str angles_option, dict angles_params,
               str dE_option, dict dE_params,
               str dz_option, dict dz_params,
               int nRays):

    cdef np.ndarray[np.float64_t, ndim=2] rays = np.zeros((nRays, 8), dtype=np.float64)
    cdef int i
    cdef double x, y, ax, ay, dE, dz
    cdef double l = 0.0
    cdef double dm = 0.0
    cdef double r_scalar, theta, phi, theta_angle, theta_max
    cdef list rays_list = []

    for i in range(nRays):
        np.random.seed(i)

        # ----- POSITION SAMPLING ----- #
        if position_option == 'circle':
            r_scalar = sqrt(np.random.uniform(0.0, 1.0))
            phi = np.random.uniform(0.0, 2 * M_PI)
            x = position_params['center'][0] + r_scalar * position_params['radius'] * cos(phi)
            y = position_params['center'][1] + r_scalar * position_params['radius'] * sin(phi)

        elif position_option == 'independent':
            x = sample_dist(position_params['x_option'], position_params['x_params'], i)
            y = sample_dist(position_params['y_option'], position_params['y_params'], i)

        else:
            raise ValueError(f"Unsupported position_option: {position_option}")

        # ----- ANGLE SAMPLING ----- #
        if angles_option == 'circle':
            r_scalar = sqrt(np.random.uniform(0.0, 1.0))
            phi = np.random.uniform(0.0, 2 * M_PI)
            ax = angles_params['center'][0] + r_scalar * angles_params['radius'] * cos(phi)
            ay = angles_params['center'][1] + r_scalar * angles_params['radius'] * sin(phi)

        elif angles_option == 'isotropic':
            theta_max = angles_params.get('theta_max', 0.25)  # radians
            u = np.random.uniform(0.0, 1.0)
            theta_angle = acos(1.0 - u * (1.0 - cos(theta_max)))
            phi = np.random.uniform(0.0, 2.0 * M_PI)
            ax = theta_angle * cos(phi)
            ay = theta_angle * sin(phi)

        elif angles_option == 'independent':
            ax = sample_dist(angles_params['ax_option'], angles_params['ax_params'], i)
            ay = sample_dist(angles_params['ay_option'], angles_params['ay_params'], i)

        else:
            raise ValueError(f"Unsupported angles_option: {angles_option}")

        # ----- ENERGY AND CHARGE SPREAD ----- #
        dE = sample_dist(dE_option, dE_params, i)
        dz = sample_dist(dz_option, dz_params, i)

        # ----- STORE RESULTS ----- #
        rays[i, 0] = x
        rays[i, 1] = ax
        rays[i, 2] = y
        rays[i, 3] = ay
        rays[i, 4] = l
        rays[i, 5] = dE
        rays[i, 6] = dm
        rays[i, 7] = dz

        rays_list.append(([x], [ax], [y], [ay], [l], [dE], [dm], [dz]))

    return rays_list

#################################################
##### HANDLING RAY PROPAGATION CALUCLATIONS #####
#################################################

# Cython function for computing terms
def compute_terms(np.ndarray[np.float64_t, ndim=1] input_array, double float_value):
    cdef Py_ssize_t length = input_array.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] output_array = np.empty(length, dtype=np.float64)
    cdef Py_ssize_t i

    for i in range(length):
        if input_array[i] == 0 and float_value == 0:
            output_array[i] = 1.0
        else:
            output_array[i] = float_value ** input_array[i]

    return output_array

# Calculates an updated ray coordinate given the corresponding COSY coordinate matrix
def parameter_transform(np.ndarray[np.float64_t, ndim=1] df_x,
                       np.ndarray[np.float64_t, ndim=1] df_ax,
                       np.ndarray[np.float64_t, ndim=1] df_y,
                       np.ndarray[np.float64_t, ndim=1] df_ay,
                       np.ndarray[np.float64_t, ndim=1] df_l,
                       np.ndarray[np.float64_t, ndim=1] df_dE,
                       np.ndarray[np.float64_t, ndim=1] df_dm,
                       np.ndarray[np.float64_t, ndim=1] df_dz,
                       np.ndarray[np.float64_t, ndim=1] df_coefficient,
                       list ray_matrices):
    cdef double x_0, ax_0, y_0, ay_0, l_0, dE_0, dm_0, dz_0

    # Extract values from the last element of ray_matrices
    cdef Py_ssize_t last_idx = len(ray_matrices) - 1
    x_0, ax_0, y_0, ay_0, l_0, dE_0, dm_0, dz_0 = [val[0] for val in ray_matrices[last_idx]]

    # Compute terms using the compute_terms function
    cdef np.ndarray[np.float64_t, ndim=1] x_term = compute_terms(df_x, x_0)
    cdef np.ndarray[np.float64_t, ndim=1] ax_term = compute_terms(df_ax, ax_0)
    cdef np.ndarray[np.float64_t, ndim=1] y_term = compute_terms(df_y, y_0)
    cdef np.ndarray[np.float64_t, ndim=1] ay_term = compute_terms(df_ay, ay_0)
    cdef np.ndarray[np.float64_t, ndim=1] l_term = compute_terms(df_l, l_0)
    cdef np.ndarray[np.float64_t, ndim=1] dE_term = compute_terms(df_dE, dE_0)
    cdef np.ndarray[np.float64_t, ndim=1] dm_term = compute_terms(df_dm, dm_0)
    cdef np.ndarray[np.float64_t, ndim=1] dz_term = compute_terms(df_dz, dz_0)

    # Convert coefficients to float64 numpy array
    coefficients = df_coefficient.astype(np.float64)

    # Compute relevant terms
    cdef np.ndarray[np.float64_t, ndim=1] relevant_terms = (coefficients * x_term * ax_term * y_term * ay_term *
                                                              l_term * dE_term * dm_term * dz_term)

    # Compute total parameter by summing the relevant terms
    cdef double total_parameter = np.sum(relevant_terms)

    return total_parameter

# Determines the updated ray coordinates after processing a full COSY matrix
def matrix_transform(list group, list ray_matrices):

    cdef list parameters = []

    for df in group:
        df_x = df['x'].values.astype(np.float64)
        df_ax = df['ax'].values.astype(np.float64)
        df_y = df['y'].values.astype(np.float64)
        df_ay = df['ay'].values.astype(np.float64)
        df_l = df['l'].values.astype(np.float64)
        df_dE = df['dE'].values.astype(np.float64)
        df_dm = df['dm'].values.astype(np.float64)
        df_dz = df['dz'].values.astype(np.float64)
        df_coefficient = df['Coefficient'].values.astype(np.float64)

        # Call the parameter_transform function with individual arrays
        new_value = parameter_transform(df_x, df_ax, df_y, df_ay, df_l, df_dE, df_dm, df_dz, df_coefficient, ray_matrices)

        parameters.append(new_value)

    new_ray = tuple([result] for result in parameters)
    #print(new_ray)
    ray_matrices.append(new_ray)

    return new_ray

# Cython function for storing coordinates
def ray_coordinates(list ray_matrices):

    cdef list x = []
    cdef list ax = []

    cdef Py_ssize_t i, length
    cdef double[:, ::1] flat_matrix
    cdef double[:] x_values, ax_values, y_values, ay_values, dE_values, dz_values

    length = len(ray_matrices)
    x_values = np.empty(length, dtype=np.float64)
    ax_values = np.empty(length, dtype=np.float64)
    y_values = np.empty(length, dtype=np.float64)
    ay_values = np.empty(length, dtype=np.float64)
    dE_values = np.empty(length, dtype=np.float64)
    dz_values = np.empty(length, dtype=np.float64)

    for i in range(length):
        flat_matrix = np.array(ray_matrices[i], dtype=np.float64)
        x_values[i] = <double>(flat_matrix[0])[0]
        ax_values[i] = <double>(flat_matrix[1])[0]
        y_values[i] = <double>(flat_matrix[2])[0]
        ay_values[i] = <double>(flat_matrix[3])[0]
        dE_values[i] = <double>(flat_matrix[5])[0]
        dz_values[i] = <double>(flat_matrix[7])[0] # Currently leaving this term out, should not change at all

    return list(x_values), list(ax_values), list(y_values), list(ay_values), list(dE_values)

#################################
##### CHECKING TRANSMISSION #####
#################################

# Process a single ray and check containment at each z using SECAR_beamline segments.
cpdef tuple process_single_ray(object ray, list grouped_data_frames, list z_positions, list polygons, list dipoleWFchanges):

    # Propagates a single ray through all provided transformation matrices
    cdef list ray_matrices = [ray]
    cdef Py_ssize_t j, k
    cdef Py_ssize_t i_last

    k = 0
    for j, group in enumerate(grouped_data_frames):
        if dipoleWFchanges != None and k<len(dipoleWFchanges) and j == dipoleWFchanges[k][0]:
          i_last = len(ray_matrices) - 1
          ray_matrices[i_last][5][0] = (ray_matrices[i_last][5][0] - 2*dipoleWFchanges[k][1]) # dE/E = 2*dB/B
        matrix_transform(group, ray_matrices)
        if dipoleWFchanges != None and k<len(dipoleWFchanges) and j == dipoleWFchanges[k][0]:
          ray_matrices[i_last+1][5][0] = (ray_matrices[i_last+1][5][0] + 2*dipoleWFchanges[k][1]) # dE/E = 2*dB/B
          k = k + 1    # Done with this dipole or WF

    # Extract coordinates before checking tranmsission
    cdef tuple coordinates = ray_coordinates(ray_matrices)
    cdef list x_values, ax_values, y_values, ay_values, dE_values
    x_values, ax_values, y_values, ay_values, dE_values = coordinates

    # Determine the point at which the ray is no longer "inside" the beamline polygons
    cdef list transmitted_x = []
    cdef list transmitted_y = []

    cdef Py_ssize_t N = len(x_values)
    cdef double x, y
    cdef object polygon
    cdef bint inside
    cdef Py_ssize_t transmission_index = -1  # -1 means the ray was blocked at the first step

    for i in range(N):
        x = x_values[i]
        y = y_values[i]
        polygon = polygons[i]

        # Skip check if no polygon is defined (open section)
        if polygon is None:
            transmitted_x.append(x)
            transmitted_y.append(y)
            transmission_index = i
            continue

        # Check if point is inside polygon
        inside = contains_xy(polygon, x, y)

        if inside:
            transmitted_x.append(x)
            transmitted_y.append(y)
            transmission_index = i
        else:
            # Stop tracking this ray after it exits the aperture
            break

    # Makes it so the transmission_index directly represents the count of transmitted points
    # (i.e., length of transmitted segments) instead of the last zero-based index.
    # This translates to the first point of failure.
    transmission_index += 1

    return (x_values, ax_values, y_values, ay_values, dE_values, transmitted_x, transmitted_y, transmission_index)

# Process all rays using precomputed beamline polygons and the above propagation calculations
cpdef process_all_rays(list rays, list grouped_data_frames, list z_positions, list polygons, list dipoleWFchanges):

    cdef list transmitted_coordinates = []

    # Loops over each ray
    for ray in rays:
        transmitted_coordinates.append(process_single_ray(ray, grouped_data_frames, z_positions, polygons, dipoleWFchanges))

    return transmitted_coordinates

#################################
##### COORDINATE EXTRACTION #####
#################################

# Define a Cython function to extract coordinate information
cpdef extract_coordinates(list transmitted_coordinates):

    # Extract coordinate information for all rays
    all_x = [transmitted_coordinate[0] for transmitted_coordinate in transmitted_coordinates]
    all_ax = [transmitted_coordinate[1] for transmitted_coordinate in transmitted_coordinates]
    all_y = [transmitted_coordinate[2] for transmitted_coordinate in transmitted_coordinates]
    all_ay = [transmitted_coordinate[3] for transmitted_coordinate in transmitted_coordinates]
    all_dE = [transmitted_coordinate[4] for transmitted_coordinate in transmitted_coordinates]

    # Specific coordinate information for rays which stay inside the beamline
    transmitted_x = [transmitted_coordinate[5] for transmitted_coordinate in transmitted_coordinates]
    transmitted_y = [transmitted_coordinate[6] for transmitted_coordinate in transmitted_coordinates]
    transmission_index = [transmitted_coordinate[7] for transmitted_coordinate in transmitted_coordinates]

    return all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_index

# Define a Cython function to sort coordinate data
# Output: [Pos. 0, Pos. 1, ..., Pos. Final] where Pos. X = [Ray 1 Value, Ray 2 Value,..., Ray N Value]
cpdef sort_coordinates(list coordinate_list):
    """
    Sorts a list of lists so that it returns a new list of lists where each sublist contains
    values at the same position for all rays. Handles cases where rays may have different lengths.

    Parameters:
    coordinate_list (list): List of lists representing coordinates for multiple rays.
                            Format: [[value 0 for ray 0, ..., value X for ray 0], ..., [value 0 for ray N, ..., value X for ray N]]

    Returns:
    list: A list of lists with the new format where each sublist contains all the values at the same position.
          Missing values are filled with `None`.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t max_length = max([len(ray) for ray in coordinate_list])
    cdef list sorted_coordinates = []
    cdef list position_group

    # Transpose the list of lists, handling different lengths
    for i in range(max_length):
        position_group = []
        for j in range(len(coordinate_list)):
            if i < len(coordinate_list[j]):
                position_group.append(coordinate_list[j][i])
            else:
                position_group.append(None)  # Fill missing data with None
        sorted_coordinates.append(position_group)

    return sorted_coordinates
