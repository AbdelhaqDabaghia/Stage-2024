# ------------------------------------------------------------------------------
# %% Preambule: imports

import os
# import numy as np

# verbose = True
verbose = False

# ------------------------------------------------------------------------------
# %% Changing directory

## Path name written incode
images_path = './'

## In the case of the path written in the shell
## python3 renaming_images.py path_name
# images_path = sys.argv[1]

os.chdir(images_path)

# ------------------------------------------------------------------------------
# %% Acquiring and sorting the gross images names

## The filenames
list_of_snapshots_names = os.listdir('./')

## Number of files
nb_files = len(list_of_snapshots_names)
print('Number of files={}'.format(nb_files))

if verbose:
    for name in list_of_snapshots_names:
        print('-'*5)
        print('Image name={}'.format(name))

## Extract the finelames characters for sorting
def time_chars(name):
    return name[-12:-4]

time_chars_list = []

# if verbose:
for name in list_of_snapshots_names:
    print('='*5)
    temp_time_chars = time_chars(name)
    print('Time characters={}'.format(temp_time_chars))
    ##
    time_chars_list.append(temp_time_chars)

## Sort the filenames list with respect to the snapshots acquisition time
list_of_sorted_snapshots_names = sorted(list_of_snapshots_names, key=time_chars)

## The order so obtained is the chronological order
for name in list_of_sorted_snapshots_names:
    print('-'*5)
    print('Image name={}'.format(name))

# ------------------------------------------------------------------------------
# %% Renaming files, with postfixes starting from 1

## int to string, add zero at the beginning of the postfix if ...
def n_to_filename_postfix(n):
    ## nb_files: list size, starting from 0
    ## n: same thing, postfix must be n+1 to avoid zero
    postfix = str(n+1)
    ## 10 to 99 images = add a zero
    if nb_files>=10 and nb_files<=99:
        if n+1<10:
            postfix = "0"+postfix
    ## 100 to 999 images = add one or two zero(s)
    if nb_files>=100 and nb_files<=999:
        # one zero
        if n+1<10:
            postfix = "00"+postfix
        # two zeros
        if n+1>=10 and n+1<100:
            postfix = "0"+postfix
    ## result
    return postfix


## Rename the images with the sorted names: png to png
for n in range(nb_files):
    ##
    temp_name = list_of_sorted_snapshots_names[n]
    new_name = 'image' + n_to_filename_postfix(n) + '.png'
    ##
    if temp_name[-4:]=='.png' :
        os.rename(temp_name, new_name)
    else:
        print('Not a png file')
