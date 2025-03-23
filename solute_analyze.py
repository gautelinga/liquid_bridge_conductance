import os
import argparse
import numpy as np
import scipy.interpolate as intp

import matplotlib.pyplot as plt

from partractools.common.utils import parse_paramsfile

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze solute")
    parser.add_argument("folder", type=str, help="Folder to look inside")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    sfolders = os.listdir(args.folder)

    print("Crawling for RandomWalkers...")
    if "RandomWalkers" in sfolders:
        print("Fixed geometry mode")

        rwfolder = os.path.join(args.folder, "RandomWalkers")

        dsets = []
        for ssfolder in os.listdir(rwfolder):
            tdatafilename = os.path.join(rwfolder, ssfolder, "0", "tdata_from_t0.000000.dat")
            paramsfilename = os.path.join(rwfolder, ssfolder, "0", "params_from_t0.000000.dat")
            
            if os.path.exists(tdatafilename) and os.path.exists(paramsfilename):
                params = parse_paramsfile(paramsfilename)
                dsets.append((tdatafilename, params))

        data_ = []

        fig, ax = plt.subplots(1, 4)
        for tdatafilename, params in dsets:
            Dm = float(params["Dm"])
            U = float(params["U"])
            dt = float(params["dt"])

            if Dm > 0 and U > 0:
                print(tdatafilename)
                tdata = np.loadtxt(tdatafilename)
                t = tdata[:, 0]
                x = tdata[:, 1]-tdata[0, 1]
                dx2 = tdata[:, 2]
                ux = tdata[:, 7]

                nt = len(t)

                popt = np.polyfit(t[nt//2:], dx2[nt//2:], 1)
                Deff = popt[0]/2

                dx2_intp = intp.InterpolatedUnivariateSpline(t, dx2)
                ddx2dt_intp = dx2_intp.derivative(1)
                ddx2dt = ddx2dt_intp(t)

                print(f"Dm={Dm}, U={U}")
                ax[0].plot(t, x/U, label=f"$D={Dm}, U={U}, dt={dt}$")
                ax[1].plot(t, dx2) #, label=f"$D={Dm}, U={U}$")
                ax[1].plot(t, 2*Deff*t)
                ax[2].plot(t, ux/U)
                ax[3].plot(t, ddx2dt)
                ax[3].plot(t, 2*Deff*np.ones_like(t))

                data_.append([Dm, U, Deff])

        for _ax in ax:
            _ax.set_xlabel(r"$t$")
            _ax.legend()

        ax[0].plot(t, t, 'k')

        ax[0].set_ylabel(r"$\langle x \rangle$")
        ax[1].set_ylabel(r"$\sigma_x^2$")
        ax[1].loglog()
        ax[2].set_ylabel(r"$\langle u_x \rangle$")
        ax[3].set_ylabel(r"$d \sigma_x^2 / dt \rangle$")
        ax[3].semilogy()

        data_ = np.array(data_)
        print(data_)
        Pe = data_[:, 1] / data_[:, 0]
        Deffn = data_[:, 2] / data_[:, 0]
        order = np.argsort(Pe)
        Pe = Pe[order]
        Deffn = Deffn[order]

        fig, ax = plt.subplots(1, 1)
        ax.scatter(Pe, Deffn)
        ax.loglog()

        ax.set_xlabel("Pe")
        ax.set_ylabel(r"$D_{eff}/D$")

        np.savetxt(os.path.join(args.folder, "RWsims.dat"), np.vstack((Pe, Deffn)).T)

        plt.show()
    else:

        fig, ax = plt.subplots(1, 1)
        for sfolder in sfolders:
            Bsimsfile = os.path.join(args.folder, sfolder, "Bsims.dat")
            print(Bsimsfile)
            if os.path.exists(Bsimsfile):
                data = np.loadtxt(Bsimsfile)
                
                ax.plot(data[:, 0], data[:, 1], label=sfolder)

        ax.legend()
        ax.loglog()
        plt.show()