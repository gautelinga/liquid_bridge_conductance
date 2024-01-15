import os
from utils import mpi_is_root, mpi_rank, mpi_size
import dolfin as df
import argparse
import meshio
import numpy as np
from utils import numpy_to_dolfin_file
import tetgen

def parse_args():
    parser = argparse.ArgumentParser(description="Mesh interior of surface meshes (.stl or similar)")
    parser.add_argument("--infolder", type=str, default="surface_meshes", help="Name of the folder containing surface meshes (.stl files)")
    parser.add_argument("--outfolder", type=str, default="volume_meshes", help="Name of the folder to store volume meshes (.h5 files)")
    parser.add_argument("--delta", type=float, default=0.05, help="Small parameter to regularize contact points between beads and plates")
    return parser.parse_args()

def generate_mesh(filename, savefile, delta=0.0, save_xdmf=True):
    m_surf = meshio.read(filename)
    nodes = m_surf.points

    if delta > 0.0:
        # Regularize contact points between beads and plates
        D = nodes[:, 2].max()
        ids_upper = np.logical_and(nodes[:, 2] > D/2, nodes[:, 2] < D - 1e-4*D)
        ids_lower = np.logical_and(nodes[:, 2] < D/2, nodes[:, 2] > 1e-4*D)
        ids_inner = nodes[:, 0]**2 + nodes[:, 1]**2 < delta**2*D**2/4
        ids_top = np.logical_and(ids_inner, ids_upper)
        ids_btm = np.logical_and(ids_inner, ids_lower)
        nodes[ids_top, 2] = D / 2 * (1 + np.sqrt(1 - delta**2)) #- 0.1*D
        nodes[ids_btm, 2] = D / 2 * (1 - np.sqrt(1 - delta**2)) #+ 0.1*D
        #m_surf.points[:] = nodes

    D = nodes[:, 2].max()
    m_surf.points /= D
    nodes = m_surf.points

    faces = m_surf.cells_dict["triangle"]

    tgen = tetgen.TetGen(nodes, faces)

    area_ = np.zeros(len(faces))
    for i, face in enumerate(faces):
        pts_loc = nodes[face, :]
        area_[i] = np.dot(pts_loc[1, :] - pts_loc[0, :], pts_loc[2, :] - pts_loc[0, :])
    dx = np.sqrt(area_.mean())
    maxvol = dx**3 #/2

    nodes, elems = tgen.tetrahedralize(plc=True, quality=True, nobisect=True, minratio=1.2, fixedvolume=True, maxvolume=maxvol)

    print("Nodes", len(nodes))
    nodes *= D
    
    #cmd = f"tetgen -pq1.1a{maxvol:1.20f}Yg {tmpfile}"
    # os.system(f"tetgen -pq1.1a0.005Yg {filename}")

    numpy_to_dolfin_file(nodes, elems, filename=savefile)

    if save_xdmf:
        mesh = df.Mesh()
        with df.HDF5File(mesh.mpi_comm(), savefile, "r") as h5f:
            h5f.read(mesh, "mesh", False)
        
        with df.XDMFFile(mesh.mpi_comm(), savefile[:-3] + "_show.xdmf") as xdmff:
            xdmff.write(mesh)

    

if __name__ == "__main__":
    args = parse_args()

    infolder = args.infolder
    outfolder = args.outfolder
    tmpfolder = "tmp"
    if mpi_is_root and not os.path.exists(outfolder):
        os.makedirs(outfolder)
    if mpi_is_root and not os.path.exists(tmpfolder):
        os.makedirs(tmpfolder)

    fnames = []
    for fname in os.listdir(infolder):
        if fname[-4:] == ".stl":
            fnames.append(fname)
    fnames = sorted(fnames)
    
    # Process each mesh in parallel
    for fname in fnames[mpi_rank::mpi_size]:
        input_fname = os.path.join(infolder, fname)
        output_fname = os.path.join(outfolder, fname[:-3] + "h5")

        if not os.path.exists(output_fname):
            print(fname)
            try:
                generate_mesh(input_fname, 
                              output_fname, save_xdmf=False)
            except:
                pass