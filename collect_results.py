import argparse
import numpy as np
from utils import mpi_max, mpi_min, mpi_print, mpi_is_root
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Solve for permeability in a single geometry")
    parser.add_argument("infolder", type=str, help="Name of folder containing .h5 volume mesh files")
    return parser.parse_args()

def key_val_split(item, delim="="):
    key, val = item.split("=")

    return key, (float(val[:-2]), val[-2:])

if __name__ == "__main__":
    args = parse_args()

    fnames = []
    for fname in os.listdir(args.infolder):
        if fname[-4:] == ".dat":
            fnames.append(fname)
    fnames = sorted(fnames)

    data_ = []
    for fname in fnames:
        prm = dict([key_val_split(item) for item in fname[:-4].split("_")])

        data_prm = [prm["Dist"][0], prm["Pc"][0], prm["Vol"][0]]
        units = dict([(key, val[1]) for key, val in prm.items()])

        data = np.loadtxt(os.path.join(args.infolder, fname))
        data_.append(np.concatenate([data_prm, data]))

    data_ = np.vstack(data_)

    dists = np.unique(data_[:, 0])

    fig, ax = plt.subplots(1, 4)
    for dist in dists:
        data_loc = data_[data_[:, 0] == dist, :]

        vol_ = data_loc[:, 2]
        k_phys_ = data_loc[:, -1]
        Pc_ = data_loc[:, 1]
        vol_phys_ = data_loc[:, 11]
        flux_bdry_rel = data_loc[:, 8] / data_loc[:, 7]

        ax[0].plot(vol_, k_phys_, '-*', label=f"dist={dist}{units['Dist']}")

        ax[1].plot(Pc_, k_phys_, '-*')    

        ax[2].plot(vol_, vol_phys_, '*')

        ax[3].plot(vol_, flux_bdry_rel, '*')
    
    ax[1].set_xlabel(f"Pressure $P_c$ [{units['Pc']}]")
    ax[1].set_ylabel(f"Conductance $k$ [a.u.]")

    ax[2].set_xlabel(f"Volume $V$ [{units['Vol']}]")
    ax[2].set_ylabel(f"Computed volume [m$^3$]")

    ax[0].set_xlabel(f"Volume [{units['Vol']}]")
    ax[0].set_ylabel(f"Conductance $k$ [a.u.]")
    ax[0].legend()

    popt = np.polyfit(data_[:, 2], data_[:, 11], 1)
    ax[2].plot(data_[:, 2], popt[0]*data_[:, 2], label=f"{popt[0]:1.2g} $V$")
    ax[2].legend()


    plt.show()