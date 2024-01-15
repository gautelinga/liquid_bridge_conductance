import os
from utils import mpi_is_root, mpi_rank, mpi_size
import argparse
import pymeshlab

def parse_args():
    parser = argparse.ArgumentParser(description="Mesh interior of surface meshes (.stl or similar)")
    parser.add_argument("--infolder", type=str, default="surface_meshes", help="Name of the folder containing surface meshes (.stl files)")
    parser.add_argument("--outfolder", type=str, default="fixed_meshes", help="Name of the folder to store fixed meshes in (.stl files)")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate meshes")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    infolder = args.infolder
    outfolder = args.outfolder
    if mpi_is_root and not os.path.exists(outfolder):
        os.makedirs(outfolder)

    fnames = []
    for fname in os.listdir(infolder):
        if fname[-4:] == ".stl":
            fnames.append(fname)
    fnames = sorted(fnames)
    
    # Process each mesh in parallel
    for fname in fnames[mpi_rank::mpi_size]:
        print(fname)

        input_fname = os.path.join(infolder, fname)
        output_fname = os.path.join(outfolder, fname)

        if not os.path.exists(output_fname) or args.regenerate:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(input_fname)
            ms.meshing_remove_null_faces()
            ms.meshing_close_holes()
            ms.meshing_isotropic_explicit_remeshing(adaptive=True, reprojectflag=False) # improve
            #ms.apply_coord_hc_laplacian_smoothing()

            ms.save_current_mesh(output_fname)