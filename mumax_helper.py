import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from scipy.signal import savgol_filter
# from matplotlib.widgets import Slider, Button


# class MumaxSimulation():
def read_mumax3_table(filename):
    """Puts the mumax3 output table in a pandas dataframe"""

    from pandas import read_table

    table = read_table(filename)
    table.columns = ' '.join(table.columns).split()[1::2]

    return table


def read_mumax3_ovffiles(outputdir):
    """Load all ovffiles in outputdir into a dictionary of numpy arrays 
    with the ovffilename (without extension) as key"""

    from subprocess import run, PIPE, STDOUT
    from glob import glob
    from os import path
    from numpy import load, where

    # convert all ovf files in the output directory to numpy files
    p = run(["mumax3-convert", "-numpy", outputdir+"/*.ovf"],
            stdout=PIPE, stderr=STDOUT)
    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    fields = []
    for npyfile in glob(outputdir+"/*.npy"):

        m = load(npyfile)

        mx = m[0, :, :, :]
        my = m[1, :, :, :]
        mz = m[2, :, :, :]

        m_magnitude = mx**2 + my**2 + mz**2

        x_coord, y_coord, z_coord = where(m_magnitude == 0.0)
        m[:, x_coord, y_coord, z_coord] = np.nan

        fields.append(m)

    return np.array(fields) 


def run_mumax3(script, name, verbose=False):
    """ Executes a mumax3 script and convert ovf files to numpy files

    Parameters
    ----------
      script:  string containing the mumax3 input script
      name:    name of the simulation (this will be the name of the script and output dir)
      verbose: print stdout of mumax3 when it is finished
    """

    from subprocess import run, PIPE, STDOUT
    from os import path, remove

    scriptfile = name + ".mx3"
    outputdir = name + ".out"

    # write the input script in scriptfile
    with open(scriptfile, 'w') as f:
        f.write(script)

    # call mumax3 to execute this script
    p = run(["mumax3", "-f", scriptfile], stdout=PIPE, stderr=STDOUT)
    if verbose or p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    if path.exists(outputdir + "/table.txt"):
        table = read_mumax3_table(outputdir + "/table.txt")
    else:
        table = None

    fields = read_mumax3_ovffiles(outputdir)
    # saves all numpy arrays in a single compressed npz file
    np.savez_compressed(outputdir + "\m.npz", *fields)

    # removes remaining ovf and npy files
    for ovfFile in glob(outputdir + "\m*.ovf"):
        remove(ovfFile)

    for npyfile in glob(outputdir + "\m*.npy"):
        remove(npyfile)

    remove(scriptfile)  # removes .mx3 file

    return fields, table


def trace_path(m_array):

    x_array = np.array([])
    y_array = np.array([])

    for time_step in m_array:

        # Very hacky: Guarentees all extremal points will be found in the same skyrmion
        # WARNING: Assumes the skyrmions initally lie along the x-axis
        if len(x_array) == 0:
            x_length = np.shape(time_step)[1]
            # cuts the image in half so only one skyrmion can be viewed
            time_step = time_step[:, :int(x_length/2)]

        y_coords, x_coords = np.unravel_index(
            np.argsort(time_step.ravel()), np.shape(time_step))

        if len(x_array) == 0:
            x_array = np.append(x_array, x_coords[0])
            y_array = np.append(y_array, y_coords[0])
        else:
            i = 0
            while np.sqrt((x_coords[i]-x_array[-1])**2 + (y_coords[i]-y_array[-1])**2) > 5:
                i += 1
            x_array = np.append(x_array, x_coords[i])
            y_array = np.append(y_array, y_coords[i])

    return savgol_filter(np.array([x_array, y_array]), 20, 2, axis=1)


def calculate_velocity(m,table, scale=1e-9):

    time_array = table['t']

    mx_pos = m[:, 0, 0, :, :]
    my_pos = m[:, 1, 0, :, :]
    mx_neg = -m[:, 0, 0, :, :]
    my_neg = -m[:, 1, 0, :, :]

    mx_pos_path = trace_path(mx_pos)
    my_pos_path = trace_path(my_pos)
    mx_neg_path = trace_path(mx_neg)
    my_neg_path = trace_path(my_neg)

    extrema_coords = np.array(
        [mx_pos_path, my_pos_path, mx_neg_path, my_neg_path])
    center_coord = np.mean(extrema_coords, axis=0)

    delta_t = np.diff(time_array)
    velocity_components = np.gradient(
        center_coord[:, 1:], delta_t[0], axis=1)*scale

    inst_velocity = np.sqrt(np.sum(velocity_components**2, axis=0))
    avg_velocity = np.mean(inst_velocity)

    return avg_velocity



