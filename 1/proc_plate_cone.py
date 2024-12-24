# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# %% Imports

import sys
# import subprocess
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import scipy
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import curve_fit

from scipy.signal import savgol_filter

from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------------------
# %% How to run the code from shell

# python3 proc_plate_cone.py "image1"

# ------------------------------------------------------------------------------
# %% The current image parameters, from shell

image_name_prefix = sys.argv[1]

## To be read with ImageJ manually

# Profile_up and profile_dn should be outside of the profile ---> 2023 10 30 experiment
profile_cone_up = int(sys.argv[4]) ## Takes into account the conic substrate
profile_cone_dn = int(sys.argv[5])

profile_bridge_up= int(sys.argv[2]) ## Truncates the bridge seeking y range, belongs to the bridge s coordinates
profile_bridge_dn= int(sys.argv[3]) ## Truncates the image to avoid the lower plate

# Profile left and profile right ...
profile_all_lf = int(sys.argv[6])
profile_all_r = int(sys.argv[7])

# parameters['paths']['image_number'] = str(int(sys.argv[1])).zfill(2)
# spline_step_in_pixel = int(sys.argv[2]) ## default = 20

profile_accuracy_bridge_cone = int(sys.argv[8])
profile_accuracy_bridge_plate = int(sys.argv[8])

# ------------------------------------------------------------------------------
# %% Global parameters

# parameters = dict()
# parameters['image'] = dict()

discr = 0.013/1794

# print('='*20)
# print('Discretization coefficient={}'.format(discr))
# print('Cubic discretization coefficient={}'.format(discr**3))
#
# profile_accuracy_bridge_cone = 3
# profile_accuracy_bridge_plate = 3 ## cause: plate-telecentric optical artefact

verbose = True
verbose = False

pixel_peradius_mm = 10
#
gamma = 0.072 ## --> Cf Mohamad, useful for the tif image ?

spline_step_in_pixel = 20

# ------------------------------------------------------------------------------
# %% For figures dimensions

figure_width_in_pixels = profile_all_r - profile_all_lf
figure_height_in_pixels = profile_cone_dn - (profile_bridge_up - 2*profile_accuracy_bridge_plate)

pixel_to_inches = discr*39.37

figure_width_in_inches = figure_width_in_pixels*pixel_to_inches
figure_height_in_inches = figure_height_in_pixels*pixel_to_inches

figure_amplification_factor = 100

# plt.figure(figsize = (figure_amplification_factor*figure_width_in_inches, figure_amplification_factor*figure_height_in_inches), dpi = 150.0)

# ------------------------------------------------------------------------------
# %% Paths towards realor processed images and other results

# parameters['pahs'] = dict()

tif_image_name = image_name_prefix+'.tif'
# path_to_tif_image = os.path.join(main_directory, tif_image_name)
path_to_tif_image = tif_image_name

# ------------------------------------------------------------------------------

results_file_name = 'processed_all_data'
temp_file_name = 'current_data'


# png_directory = os.path.join(main_directory, 'png')
png_directory = 'png'
if not os.path.isdir(png_directory):
    os.mkdir(png_directory)

## For the txt data frame generated from during the image processing
# text_directory = os.path.join(main_directory, 'txt')
text_directory = 'txt'
if not os.path.isdir(text_directory):
    os.mkdir(text_directory)

## For a single image: single line table
path_to_results_file_name = os.path.join(text_directory, results_file_name)

## A text file to be written, in order not to overwrite the results file name if it exists
path_to_temp_file_name = os.path.join(text_directory, temp_file_name)

# ------------------------------------------------------------------------------
# %% Opening the current image

I = plt.imread(path_to_tif_image)

print('-'*20)
if verbose == True:
    print('I={}'.format(I))
    print('-'*20)
    print('I shape={}'.format(np.shape(I)))
    print('-'*20)

# ------------------------------------------------------------------------------
# %% Boucle pour la construction de la generatrice de Omega_lg

def y_to_x(y, side="left"):

    if side == "left":
        x = profile_all_lf
    elif side == "right":
        x = profile_all_r

    ## Initialization, the true values lies in the while loop
    x_neighbor = 0

    cont = True

    while cont:

        if side == "left":
            x_neighbor = x+1
        elif side == "right":
            x_neighbor = x-1

        ## For both sides
        if I[y, x] != I[y, x_neighbor]:
            cont = False
        else:
            x = x_neighbor

    return x

# ------------------------------------------------------------------------------

y_array = np.arange(profile_bridge_up, profile_cone_dn)

# ------------------------------------------------------------------------------

x_from_y_left = np.ones(profile_cone_dn - profile_bridge_up)

for y_index in range(len(y_array)):
    x = y_array[y_index]
    x_from_y_left[y_index] = y_to_x(x)

    # if verbose == True and y_index%400 == 0:
    #     print('-'*10)
    #     print('Saut effectue a y={}'.format(x_from_y_right[y_index]))
    #     print('-'*10)

if verbose == True:
    print(x_from_y_left)
    print('-'*10)

# ------------------------------------------------------------------------------

x_from_y_right = np.ones(profile_cone_dn - profile_bridge_up)

for y_index in range(len(y_array)):
    x = y_array[y_index]
    x_from_y_right[y_index] = y_to_x(x, "right")

    # if verbose == True and y_index%400 == 0:
    #     print('-'*10)
    #     print('Saut effectue a y={}'.format(x_from_y_right[y_index]))
    #     print('-'*10)

if verbose == True:
    print(x_from_y_right)
    print('-'*10)

# ------------------------------------------------------------------------------

fig_brute_data = plt.figure(figsize = (figure_amplification_factor*figure_width_in_inches, figure_amplification_factor*figure_height_in_inches), dpi = 150.0)

# A = mpimg.imread(os.path.join(main_directory, image_name_prefix)+'.tif')
A = mpimg.imread(image_name_prefix+'.tif')
A1 = A[profile_bridge_up:profile_cone_dn, profile_all_lf:profile_all_r]
A_graph = np.flip(A1, axis=0)
plt.imshow(A_graph, cmap='gray', extent=(profile_all_lf, profile_all_r, profile_bridge_up, profile_cone_dn), aspect = 'equal')

plt.xlim(profile_all_lf, profile_all_r)
plt.ylim(profile_bridge_up, profile_cone_dn)

plt.axis('equal')
plt.axis('tight')

plt.plot(x_from_y_left, y_array, 'r-', markersize=5, label = 'left boundary')
plt.plot(x_from_y_right, y_array, 'c-', markersize=5, label = 'right boundary')
plt.gca().invert_yaxis()
plt.legend(loc = 'best', fontsize = 'large')
plt.title('Pixelized boundaries', size = 13)
plt.savefig(os.path.join(png_directory, image_name_prefix+'_left_right_boundaries.png'))
plt.show()
plt.close()

# ------------------------------------------------------------------------------
# %% Linear regression for the cone: uses profile_cone_up and profile_cone_dn

## y_array
## x_from_y_left
## x_from_y_right

y_cone_updn_indexes = np.where(y_array >= profile_cone_up) ## y_array >= profile_cone_up //

y_array_restricted_to_cone = y_array[(y_cone_updn_indexes)[0][0]:(y_cone_updn_indexes)[0][-1]].reshape((-1, 1))

x_left_restricted_to_cone = x_from_y_left[(y_cone_updn_indexes)[0][0]:(y_cone_updn_indexes)[0][-1]]
x_right_restricted_to_cone = x_from_y_right[(y_cone_updn_indexes)[0][0]:(y_cone_updn_indexes)[0][-1]]

if verbose:
    print('%'*50)
    print(y_cone_updn_indexes)
    print('-'*20)
    print(y_array_restricted_to_cone)
    print(x_left_restricted_to_cone)
    print(x_right_restricted_to_cone)

model_left = LinearRegression()
model_right = LinearRegression()

model_left.fit(y_array_restricted_to_cone, x_left_restricted_to_cone)
model_right.fit(y_array_restricted_to_cone, x_right_restricted_to_cone )

if verbose:
    print('><'*10)
    print('R left', model_left.score(y_array_restricted_to_cone, x_left_restricted_to_cone))
    print('R right', model_right.score(y_array_restricted_to_cone, x_right_restricted_to_cone))
    print('-'*10)
    print('left intercept', model_left.intercept_)
    print('left slope', model_left.coef_)
    print('-'*10)
    print('right intercept', model_right.intercept_)
    print('right slope', model_right.coef_)

## Opening angles: using the regression
beta_solid_left = np.arctan(model_left.coef_)[0]
beta_solid_right = np.arctan(model_right.coef_)[0]

## Recover the paper notation
alpha_left = -beta_solid_left
alpha_right = beta_solid_right

print('%'*20)
print('LOWER TRIPLE POINTS')
print('='*10)
print('Left derivative on solid phase', model_left.coef_)
print('Right derivative on solid phase', model_right.coef_)
print('='*10)
print('alpha left={} radians'.format(alpha_left))
print('alpha right={} radians'.format(alpha_right))
print('-'*20)

# ------------------------------------------------------------------------------
# %% Solid and liquid components identifications through the linear regression validity

y_regression_indexes_left = np.where((model_left.predict(y_array.reshape((-1, 1))) - x_from_y_left)**2 < profile_accuracy_bridge_cone**2)[0]
y_regression_indexes_right = np.where((model_right.predict(y_array.reshape((-1, 1))) - x_from_y_right)**2 < profile_accuracy_bridge_cone**2)[0]

y_regression_left_min_index = y_regression_indexes_left[0]
y_regression_right_min_index = y_regression_indexes_right[0]

if verbose:
    print('y_regression_left_min_index', y_regression_left_min_index)
    print('y_regression_right_min_index', y_regression_right_min_index)

## The greatest pixel height so that all indexes above belong to the cone or are close to the triple points
y_regression_updn_min_index = min(y_regression_left_min_index, y_regression_right_min_index)

## Gets the bridge profile dn y index
y_bridge_lower_shell_index = np.where(y_array == profile_bridge_dn)[0]

# Compare les valeurs et choisit la plus petite
if isinstance(y_bridge_lower_shell_index, np.ndarray) and len(y_bridge_lower_shell_index) > 0:
    y_bridge_max_index = min(y_regression_updn_min_index, y_bridge_lower_shell_index.min())
else:
    y_bridge_max_index = y_regression_updn_min_index

if verbose:
    print('Maximum possibly y index')
    print('With regression only', y_regression_updn_min_index)
    print('From the bridge upper profile', y_bridge_max_index)
    print('-'*10)

fig_three_components = plt.figure(figsize = (figure_amplification_factor*figure_width_in_inches, figure_amplification_factor*figure_height_in_inches), dpi = 150.0)

# A = mpimg.imread(os.path.join(main_directory, image_name_prefix)+'.tif')
A = mpimg.imread(image_name_prefix+'.tif')
A1 = A[profile_bridge_up:profile_cone_dn, profile_all_lf:profile_all_r]
A_graph = np.flip(A1, axis=0)
plt.imshow(A_graph, cmap='gray', extent=(profile_all_lf, profile_all_r, profile_bridge_up, profile_cone_dn), aspect = 'equal')

plt.xlim(profile_all_lf, profile_all_r)
plt.ylim(profile_bridge_up, profile_cone_dn)

plt.axis('equal')
plt.axis('tight')

plt.plot(x_from_y_left[y_regression_updn_min_index:], y_array[y_regression_updn_min_index:], 'y-', markersize=5, label = 'left cone')
plt.plot(x_from_y_left[y_bridge_max_index:y_regression_updn_min_index], y_array[y_bridge_max_index:y_regression_updn_min_index], 'b-', markersize=5, label = 'undetermined')
plt.plot(x_from_y_left[:y_bridge_max_index], y_array[:y_bridge_max_index], 'r-', markersize=5, label = 'left bridge')
##
plt.plot(x_from_y_right[y_regression_updn_min_index:], y_array[y_regression_updn_min_index:], 'y-', markersize=5, label = 'right cone')
plt.plot(x_from_y_right[y_bridge_max_index:y_regression_updn_min_index], y_array[y_bridge_max_index:y_regression_updn_min_index], 'b-', markersize=5, label = 'undetermined')
plt.plot(x_from_y_right[:y_bridge_max_index], y_array[:y_bridge_max_index], 'r-', markersize=5, label = 'right bridge')
##
plt.gca().invert_yaxis()
# plt.legend(loc = 'best', fontsize = 'large')
plt.title('Two sides, three components', size = 13)
plt.savefig(os.path.join(png_directory, image_name_prefix+'_left_right_cone_fit_bridge.png'))
plt.show()
plt.close()

# ------------------------------------------------------------------------------
# %% The triple points

triple_point_lower_left = [x_from_y_left[y_regression_left_min_index], y_array[y_regression_left_min_index]]
triple_point_lower_right = [x_from_y_right[y_regression_right_min_index], y_array[y_regression_right_min_index]]

if verbose:
    print('triple point cone left={} pixels'.format(triple_point_lower_left))
    print('triple point cone right={} pixels'.format(triple_point_lower_right))
    print('%'*20)

## Definition of upper triple points: with the input in shell, prior to the code call
triple_point_upper_left = [x_from_y_left[0], profile_bridge_up]
triple_point_upper_right = [x_from_y_right[0], profile_bridge_up]

if verbose:
    print('%'*20)
    print('UPPER TRIPLE POINTS')
    print('='*10)
    print('triple point plate left={} pixels'.format(triple_point_upper_left))
    print('triple point plate right={} pixels'.format(triple_point_upper_right))
print('%'*20)

## Twice yc at the top of the bridge
diam_cone_bridge = triple_point_lower_right[0] - triple_point_lower_left[0]

## Twice yc at the top of the bridge
diam_plate_bridge = triple_point_upper_right[0] - triple_point_upper_left[0]

yc_cone_bridge = 0.5*diam_cone_bridge
yc_plate_bridge = 0.5*diam_plate_bridge

print('%'*50)
if verbose:
    print('yc={} pixels'.format(abs(yc_cone_bridge)))
    print('-'*10)
print('yc={} m'.format(abs(yc_cone_bridge)*discr))
print('yc={} mm'.format(abs(yc_cone_bridge)*1000*discr))
print('='*50)
print('yl={} pixels'.format(abs(yc_plate_bridge)))
print('-'*10)
print('yl={} m'.format(abs(yc_plate_bridge)*discr))
print('yl={} mm'.format(abs(yc_plate_bridge)*1000*discr))
print('%'*50)

# ------------------------------------------------------------------------------
# %% Bridge edges processing: 6th degree polynomial regression

y_bridge = y_array[:y_bridge_max_index]

y_left_bridge = y_array[:y_regression_left_min_index]
y_right_bridge = y_array[:y_regression_right_min_index]

x_left_bridge = x_from_y_left[:y_regression_left_min_index]
x_right_bridge = x_from_y_right[:y_regression_right_min_index]

poly_fit_left_coefficients = np.polyfit(y_left_bridge, x_left_bridge, 6)
poly_fit_right_coefficients = np.polyfit(y_right_bridge, x_right_bridge, 6)

polynomial_left = np.poly1d(poly_fit_left_coefficients)
polynomial_right = np.poly1d(poly_fit_right_coefficients)

# ------------------------------------------------------------------------------

derivative_cone_left = np.polyder(polynomial_left)(y_array[y_regression_left_min_index])
derivative_cone_right = np.polyder(polynomial_right)(y_array[y_regression_right_min_index])

beta_liquid_cone_left = np.arctan(derivative_cone_left)#[0]
beta_liquid_cone_right = np.arctan(derivative_cone_right)#[0]

# ------------------------------------------------------------------------------

derivative_plate_left = np.polyder(polynomial_left)(y_array[0])
derivative_plate_right = np.polyder(polynomial_right)(y_array[0])

beta_liquid_plate_left = np.arctan(derivative_plate_left)#[0]
beta_liquid_plate_right = np.arctan(derivative_plate_right)#[0]

if verbose:
    print('-'*10)
    print('Left derivative on liquid phase', derivative_cone_left)
    print('Right derivative on liquid phase', derivative_cone_right)
    print('%'*20)
    print('Left bridge edge angle to vertical={} radians'.format(beta_liquid_cone_left))
    print('Right bridge edge angle to vertical={} radians'.format(beta_liquid_cone_right))

# ------------------------------------------------------------------------------
# %% y_star: extreme widths, vanishing derivative

## Extreme widths
## The height of the neck
y_min_star_index = np.argmin(x_from_y_right[:y_bridge_max_index] - x_from_y_left[:y_bridge_max_index])
height_star_argmin = y_array[y_min_star_index]
x_left_min_width = x_from_y_left[y_min_star_index]
x_right_min_width = x_from_y_right[y_min_star_index]
## The height of the gorge
y_max_star_index = np.argmax(x_from_y_right[:y_bridge_max_index] - x_from_y_left[:y_bridge_max_index])
height_star_argmax = y_array[y_max_star_index]
x_left_max_width = x_from_y_left[y_max_star_index]
x_right_max_width = x_from_y_right[y_max_star_index]

# ------------------------------------------------------------------------------

zero_polynomial_left = np.polyder(polynomial_left).r
zero_polynomial_right = np.polyder(polynomial_right).r

isreal_zero_left = np.isreal(zero_polynomial_left)
isreal_zero_right = np.isreal(zero_polynomial_right)

bridge_zero_list_left = []
bridge_zero_list_right = []

for i in range(len(zero_polynomial_left)):
    if isreal_zero_left[i] == True:
        zero_ordinate = int(np.real(zero_polynomial_left[i]))
        if zero_ordinate > y_array[0] and zero_ordinate < y_array[y_bridge_max_index]:
            bridge_zero_list_left.append(zero_ordinate)

for i in range(len(zero_polynomial_right)):
    if isreal_zero_right[i] == True:
        zero_ordinate = int(np.real(zero_polynomial_right[i]))
        if zero_ordinate > y_array[0] and zero_ordinate < y_array[y_bridge_max_index]:
            bridge_zero_list_right.append(zero_ordinate)

print('%'*20)
print('Minimum width at {} pixels'.format(height_star_argmin))
print('-'*10)
print('Maximum width at {} pixels'.format(height_star_argmax))
print('='*10)
print('List of singular ordinates for the left side of the bridge={}'.format(bridge_zero_list_left))
print('List of singular ordinates for the right side of the bridge={}'.format(bridge_zero_list_right))
print('x'*30)

local_extremum_exists = len(bridge_zero_list_left) > 0 and len(bridge_zero_list_left) > 0

if local_extremum_exists:
    if bridge_zero_list_left and bridge_zero_list_right:  # Vérifie que les listes ne sont pas vides
        pixel_number = int(0.5 * (bridge_zero_list_left[0] + bridge_zero_list_right[0]))
        y_star_index = np.where(y_array == pixel_number)
    else:
        print("Les listes bridge_zero_list_left ou bridge_zero_list_right sont vides.")


## y: with the convention of Millet et al. papers
y_star_left = x_from_y_left[y_min_star_index]
y_star_right = x_from_y_right[y_min_star_index]

print('Right minimum width at={} pixels'.format(zero_polynomial_left))
print('-'*10)

if local_extremum_exists:
    print('x'*30)
    if isinstance(y_star_right, (list, tuple)) and isinstance(y_star_left, (list, tuple)):
        print('Local extremum exists with y star={} pixels'.format(y_star_right - y_star_left))
        print('-'*10)
        print('y star={} m'.format(discr*(y_star_right - y_star_left)))
        print('y star={} mm'.format(1000*discr*(y_star_right - y_star_left)))
        print('x'*30)
    else:
        print('Error: y_star_right and y_star_left should be lists or tuples.')
        print('!'*30)
else:
    print('!'*30)
    print('NO local extremum for the bridge profile')
    print('!'*30)

# --------------------------------------------------------------------
# %% Deduce the contact angles at the cone and the plate

theta_cone_left = abs(beta_solid_left - beta_liquid_cone_left)
theta_cone_right = abs(beta_solid_right - beta_liquid_cone_right)

theta_plate_left = abs(np.pi/2 - beta_liquid_plate_left)
theta_plate_right = abs(np.pi/2 - beta_liquid_plate_right)

print('%'*100)
print('CONTACT ANGLES')
print('='*20)
if verbose:
    print('theta cone left={} radians'.format(theta_cone_left))
    print('theta cone right={} radians'.format(theta_cone_right))
    print('-'*20)
print('theta cone left={} degres'.format(theta_cone_left*180/np.pi))
print('theta cone right={} degres'.format(theta_cone_right*180/np.pi))
print('='*20)
if verbose:
    print('theta plate left={} radians'.format(theta_plate_left))
    print('theta plate right={} radians'.format(theta_plate_right))
    print('-'*20)
print('theta plate left={} degres'.format(theta_plate_left*180/np.pi))
print('theta plate right={} degres'.format(theta_plate_right*180/np.pi))


# ------------------------------------------------------------------------------
# %% Overall figure

fig_reg = plt.figure(figsize = (figure_amplification_factor*figure_width_in_inches, figure_amplification_factor*figure_height_in_inches), dpi = 150.0)

# A = mpimg.imread(os.path.join(main_directory, image_name_prefix)+'.tif')
A = mpimg.imread(image_name_prefix+'.tif')
# A1 = A[profile_bridge_up:profile_cone_dn, profile_all_lf:profile_all_r]
A1 = A[(profile_bridge_up - 2*profile_accuracy_bridge_plate):profile_cone_dn, profile_all_lf:profile_all_r]
A_graph = np.flip(A1, axis=0)
# plt.imshow(A_graph, cmap='gray', extent=(profile_all_lf, profile_all_r, profile_bridge_up, profile_cone_dn), aspect = 'equal')
plt.imshow(A_graph, cmap='gray', extent=(profile_all_lf, profile_all_r, (profile_bridge_up - 2*profile_accuracy_bridge_plate), profile_cone_dn), aspect = 'equal')

plt.xlim(profile_all_lf, profile_all_r)
plt.ylim((profile_bridge_up - 2*profile_accuracy_bridge_plate), profile_cone_dn)

plt.axis('equal')
plt.axis('tight')

plt.plot(model_left.predict(y_array[y_regression_left_min_index:].reshape((-1, 1))), y_array[y_regression_left_min_index:], 'g-', markersize=5, label = 'left cone')
plt.plot(np.polyval(poly_fit_left_coefficients, y_left_bridge), y_left_bridge, 'r-', markersize=5, label = 'left bridge')
##
plt.plot(model_right.predict(y_array[y_regression_right_min_index:].reshape((-1, 1))), y_array[y_regression_right_min_index:], 'c-', markersize=5, label = 'right cone')
plt.plot(np.polyval(poly_fit_right_coefficients, y_right_bridge), y_right_bridge, 'b-', markersize=5, label = 'right bridge')
##
## Plot the triple points, localized automatically
plt.plot(triple_point_upper_left[0], triple_point_upper_left[1], 'cx', markersize=7, linewidth=3)
plt.plot(triple_point_upper_right[0], triple_point_upper_right[1], 'cx', markersize=7, linewidth=3)
plt.plot(triple_point_lower_left[0], triple_point_lower_left[1], 'gx', markersize=7, linewidth=3)
plt.plot(triple_point_lower_right[0], triple_point_lower_right[1], 'gx', markersize=7, linewidth=3)
## Plot the points for minimum width and the line between them
plt.plot(x_left_min_width, height_star_argmin, 'yx', markersize=7, linewidth=3)
plt.plot(x_right_min_width, height_star_argmin, 'yx', markersize=7, linewidth=3)
plt.plot(np.linspace(x_left_min_width, x_right_min_width, 100), height_star_argmin*np.ones(100), '--y', markersize=2)
## Plot the points for maximum width and the line between them
plt.plot(x_left_max_width, height_star_argmax, 'y+', markersize=7, linewidth=3)
plt.plot(x_right_max_width, height_star_argmax, 'y+', markersize=7, linewidth=3)
plt.plot(np.linspace(x_left_max_width, x_right_max_width, 100), height_star_argmax*np.ones(100), '.y', markersize=2)
## Plot, if they exist, the points at the neck or the gorge
if local_extremum_exists:
    plt.plot(y_star_left, bridge_zero_list_left[0], 'yo', markersize=7, linewidth=3)
    plt.plot(y_star_right, bridge_zero_list_right[0], 'yo', markersize=7, linewidth=3)
    plt.plot(np.linspace(y_star_left, y_star_right, 100), pixel_number*np.ones(100), '-y', markersize=1)
    #
##
plt.gca().invert_yaxis()
plt.legend(loc = 'best', fontsize = 'large')
plt.title('Two sides, two regressions', size = 13)
plt.savefig(os.path.join(png_directory, image_name_prefix+'_left_right_cone_bridge_regressions.png'))
plt.show()
plt.close()



# %% Saving data: possibly several images

temp_file = open(path_to_temp_file_name + '.txt', 'w')

en_tete = 'tif_image_name' + '\t' + 'bridge_up (y)' + '\t' + 'bridge_dn (y)' + '\t' + 'profil_cone_up (y)' + '\t' + 'profil_cone_dn (y)' + '\t' + 'profil_left (x)' + '\t' + 'profil_right (x)' + '\t' + 'alpha_left' + '\t' + 'alpha_right' + '\t' + 'yc' + '\t' + 'yl (mm)' + '\t'  + '\t' + 'theta cone left' + '\t' + 'theta cone right' + '\t' + 'theta plate left' + '\t' + 'theta plate right' + '\t' + 'theta_cone_moy' + '\t' + 'theta_plate_moy' + '\t' + 'alpha_moy' + '\t' + 'yl (ml)\n'

if os.path.isfile(path_to_results_file_name + '.txt'):
    results_file = open(path_to_results_file_name + '.txt', 'r')
    for line in results_file:
        ## The line is written only if it doesn't contain the current image data. Thus the results file is overwritten.
        temp_line_contents = line.split(' ')
        head_name = temp_line_contents[0]
        print('head name', head_name)
        if head_name != image_name_prefix:
            temp_file.write(line)
        else:
            print('Results are being overwritten')
else:
    ## No data written yet in the temp file
    temp_file.write(en_tete)

## A single line of data to fill
temp_file.write(image_name_prefix + '\t' + str(profile_cone_up) + '\t' + str(profile_cone_dn) + '\t' + str(profile_bridge_up) + '\t' + str(profile_bridge_dn) + '\t' + str(profile_all_lf) + '\t' + str(profile_all_r) + '\t' + str(alpha_left) + '\t' + str(alpha_right) + '\t' + str(abs(yc_plate_bridge)*1000*discr) + '\t' + str(abs(yc_cone_bridge)*1000*discr) + '\t' + '\t' + str(format(theta_cone_left*180/np.pi)) + '\t' + str(format(theta_cone_right*180/np.pi)) + '\t' + str(format(theta_plate_left*180/np.pi)) + '\t' + str(format(theta_plate_right*180/np.pi)))

temp_file.close()

if os.path.isfile(path_to_results_file_name + '.txt'):
    results_file.close()

# Vérifier si le fichier de résultats final existe déjà
if os.path.isfile(path_to_results_file_name + '.txt'):
    sys.exit('Le fichier de résultats final existe déjà. Le processus est arrêté.')

# Renommer le fichier temporaire en fichier de résultats final
os.rename(path_to_temp_file_name + '.txt', path_to_results_file_name + '.txt')

# Sortir avec un message indiquant que le code doit être réécrit
sys.exit('%'*30 + ' Code à réécrire ' + '%'*30)











# ------------------------------------------------------------------------------
# %%
