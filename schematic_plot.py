import os
from utils import mpi_is_root, mpi_rank, mpi_size, key_val_split
import argparse
import dolfin as df
import meshio
import matplotlib.pyplot as plt
import numpy as np
from geometry_analyze import make_edge_dict, get_submesh, reorient_faces

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import igl
import scipy as sp
from scipy import optimize as opt

#import matplotlib.font_manager
#aa = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#print(aa)


plt.rcParams.update({
    "grid.color": "#eeeeee",
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial"
})

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze geometries")
    #parser.add_argument("--meshfolder", type=str, required=True, help="Name of the folder containing mesh files (.h5).")
    parser.add_argument("--datafolder", type=str, required=True, help="Name of the folder containing data files (.xdmf).")
    parser.add_argument("--show", action="store_true", help="Show plots")
    return parser.parse_args()

def get_perimeter(faces):
    e = make_edge_dict(faces)
    v2v = dict()
    for key, val in e.items():
        if len(val) == 1:
            v1, v2 = key
            if v1 in v2v:
                v2v[v1].add(v2)
            else:
                v2v[v1] = set([v2])
            if v2 in v2v:
                v2v[v2].add(v1)
            else:
                v2v[v2] = set([v1])
    
    v__ = []
    while len(v2v) > 0:
        v0 = list(v2v.keys())[0]
        vp = v0

        v = list(v2v.pop(vp))[0]
        v_ = [v0]

        while v != v0:
            v_.append(v)
            vn = (v2v.pop(v) - set([vp])).pop()
            #print(v, v0, vn)
            vp = v
            v = vn

        v_ = np.array(v_)
        v__.append(v_)
    return v__


def add_poly(ax, xx, edgecolors, facecolors, linewidths, alpha):
    verts = [list(zip(xx[:, 0], xx[:, 1], xx[:, 2]))]
    ax.add_collection3d(Poly3DCollection(verts, edgecolors=edgecolors, facecolors=facecolors, linewidths=linewidths, alpha=alpha, zsort="min"))


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


if __name__ == "__main__":
    args = parse_args()

    datafolder = args.datafolder
    #meshfolder = args.meshfolder

    fnames = []
    for fname in os.listdir(datafolder):
        if fname[-10:] == "_subd.xdmf": # and os.path.exists(os.path.join(meshfolder, fname[:-8] + ".h5")):
            fnames.append(fname[:-10])
    fnames = sorted(fnames)
    
    data_ = []

    # define origin and axes
    o = np.array([0,0,0])
    x0 = np.array([1,0,0])
    y0 = np.array([0,1,0])
    z0 = np.array([0,0,1])

    alpha = 0.8
    alpha2 = 0.4
    gray = "#dedede"
    graydark = "#cfcfcf"
    graydarker = "#c0c0c0"
    gray2 = "#cecece"
    graylight = "#eeeeee"
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0, zorder=100)
    text_options = {'horizontalalignment': 'center',
                    'verticalalignment': 'center',
                    'fontsize': 14, 
                    "zorder": 100}
    
    # Process each mesh in parallel
    for fname in fnames:
        print(fname)
        prm = dict([key_val_split(item) for item in fname[:-4].split("_")])

        #mesh_fname = os.path.join(meshfolder, fname + ".h5")
        subd_fname = os.path.join(datafolder, fname + "_subd.xdmf")
        output_fname = os.path.join(datafolder, fname + "_schematic_figure.pdf")

        m = meshio.read(subd_fname)

        nodes = m.points
        faces = m.cells_dict["triangle"]
        subd = m.cell_data["subd"][0].flatten()

        exterior_faces = subd > 0
        faces = faces[exterior_faces, :]
        subd = subd[exterior_faces]

        nodes, faces = reorient_faces(nodes, faces)

        # Inlet
        nodes_inlet, faces_inlet = get_submesh(nodes, faces, subd == 3)
        nodes_outlet, faces_outlet = get_submesh(nodes, faces, subd == 4)
        nodes_bdry, faces_bdry = get_submesh(nodes, faces, subd == 7)
        nodes_bead, faces_bead = get_submesh(nodes, faces, subd == 6)
        nodes_mirr, faces_mirr = get_submesh(nodes, faces, subd == 5)
        nodes_top, faces_top = get_submesh(nodes, faces, subd == 1)
        nodes_btm, faces_btm = get_submesh(nodes, faces, subd == 2)

        v_bdry_ = get_perimeter(faces_bdry)
        v_bead_ = get_perimeter(faces_bead)
        v_inlet_ = get_perimeter(faces_inlet)
        v_outlet_ = get_perimeter(faces_outlet)
        v_mirr_ = get_perimeter(faces_mirr)
        v_top_ = get_perimeter(faces_top)
        v_btm_ = get_perimeter(faces_btm)

        xmax = 0.
        ymax = 0.
        
        ax = plt.axes(projection='3d', computed_zorder=True)
        fig = ax.get_figure()

        for v_ in v_bdry_:
            xx = nodes_bdry[v_, :]
            xx[:, 2] -= 0.5
            xmax = max(xmax, xx[:, 0].max())
            ymax = max(ymax, xx[:, 1].max())

            add_poly(ax, xx, 'k', graydark, 1, alpha)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, graydark, 1, alpha2)
            xx[:, 0] = 2*xmax-xx[:, 0]
            add_poly(ax, xx, gray2, graydark, 1, alpha2)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, graydark, 1, alpha2)

        for v_ in v_inlet_:
            xx = nodes_inlet[v_, :]
            xx[:, 2] -= 0.5

            add_poly(ax, xx, 'k', gray, 1, alpha)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 0] = 2*xmax-xx[:, 0]
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)

        for v_ in v_outlet_:
            xx = nodes_outlet[v_, :]
            xx[:, 2] -= 0.5

            add_poly(ax, xx, 'k', gray, 1, alpha)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 0] = 2*xmax-xx[:, 0]
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)

        for v_ in v_mirr_:
            xx = nodes_mirr[v_, :]
            xx[:, 2] -= 0.5

            add_poly(ax, xx, 'k', gray, 1, alpha)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 0] = 2*xmax-xx[:, 0]
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)

        for v_ in v_top_:
            xx = nodes_top[v_, :]
            xx[:, 2] -= 0.5

            add_poly(ax, xx, 'k', graylight, 1, alpha)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, graylight, 1, alpha2)
            xx[:, 0] = 2*xmax-xx[:, 0]
            add_poly(ax, xx, gray2, graylight, 1, alpha2)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, graylight, 1, alpha2)

        for v_ in v_btm_:
            xx = nodes_btm[v_, :]
            xx[:, 2] -= 0.5

            add_poly(ax, xx, 'k', graydarker, 1, alpha)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, graydarker, 1, alpha2)
            xx[:, 0] = 2*xmax-xx[:, 0]
            add_poly(ax, xx, gray2, graydarker, 1, alpha2)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, graydarker, 1, alpha2)

        for v_ in v_bead_:
            xx = nodes_bead[v_, :]
            xx[:, 2] -= 0.5

            add_poly(ax, xx, 'k', gray, 1, alpha)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 0] = 2*xmax-xx[:, 0]
            add_poly(ax, xx, gray2, gray, 1, alpha2)
            xx[:, 1] *= -1
            add_poly(ax, xx, gray2, gray, 1, alpha2)

        ymax = max(ymax, 0.5)

        arrow_size = 0.2
        factor = 1.2

        if False:
            ex = Arrow3D([o[0], arrow_size*x0[0]], [o[1], arrow_size*x0[1]], [o[2], arrow_size*x0[2]], **arrow_prop_dict)
            ey = Arrow3D([o[0], arrow_size*y0[0]], [o[1], arrow_size*y0[1]], [o[2], arrow_size*y0[2]], **arrow_prop_dict)
            ez = Arrow3D([o[0], arrow_size*z0[0]], [o[1], arrow_size*z0[1]], [o[2], arrow_size*z0[2]], **arrow_prop_dict)
            ex.set_zorder(100)
            ax.add_artist(ex)
            ex.set_zorder(100)
            ax.add_artist(ey)
            ax.add_artist(ez)


            # add labels for x axes
            ax.text(factor*arrow_size*x0[0], factor*arrow_size*x0[1], factor*arrow_size*x0[2],r'$x$', **text_options)
            ax.text(factor*arrow_size*y0[0], factor*arrow_size*y0[1], factor*arrow_size*y0[2],r'$y$', **text_options)
            ax.text(factor*arrow_size*z0[0], factor*arrow_size*z0[1], factor*arrow_size*z0[2],r'$z$', **text_options)


        ax.set_xlim(0, 2*xmax)
        #ax.set_ylim(0-ymax, ymax)
        ax.set_ylim(-ymax, ymax)
        ax.set_zlim(-0.5, 0.5)
        ax.set_box_aspect(aspect=(2*xmax, 2*ymax, 1))

        # add label for origin
        #ax.text(0.0, 0.0, -0.05,r'$o$', **text_options)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")

        ax.view_init(elev=30, azim=135, roll=0)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        ax.xaxis.set_ticks([0, xmax, 2*xmax], labels=[r"0", r"$\ell/2$", r"$\ell$"])
        ax.yaxis.set_ticks([-0.5, 0, 0.5], labels=["$-d/2$", "0", "$d/2$"])
        ax.zaxis.set_ticks([-0.5, 0, 0.5], labels=["$-d/2$", "0", "$d/2$"])

        #ax.grid(True, color="#eeeeee")
        #ax.grid(False)
        #ax.grid(c="red", linestyle="--")

        #ax.xaxis.gridlines.set_lw(3.0)
        #ax.yaxis.gridlines.set_lw(3.0)
        #ax.zaxis.gridlines.set_lw(3.0)
        #print(ax.zaxis._axinfo)
        #.update({'grid' : {'color': (0, 0, 0, 1)}})

        #plt.tight_layout()

        fig.savefig(output_fname)

        if args.show:
            plt.show()