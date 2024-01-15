import os
from utils import mpi_is_root, mpi_rank, mpi_size, numpy_to_dolfin_file, key_val_split
import argparse
import dolfin as df
import meshio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
import trimesh

import igl
import scipy as sp
from scipy import optimize as opt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze geometries")
    #parser.add_argument("--meshfolder", type=str, required=True, help="Name of the folder containing mesh files (.h5).")
    parser.add_argument("--datafolder", type=str, required=True, help="Name of the folder containing data files (.xdmf).")
    parser.add_argument("--show", action="store_true", help="Show plots")
    return parser.parse_args()

def get_submesh(nodes, faces, ids):
    faces_sub = faces[ids, :]
    old_node_ids = np.unique(faces_sub)
    old_to_new_node_ids = np.zeros(len(nodes), dtype=int)
    for new_node_id, old_node_id in enumerate(old_node_ids):
        old_to_new_node_ids[old_node_id] = new_node_id
    nodes_sub = nodes[old_node_ids, :]
    faces_sub = old_to_new_node_ids[faces_sub]
    return nodes_sub, faces_sub

def compute_mesh_area(mesh_fname):
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), mesh_fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    return df.assemble(df.Constant(1.) * df.dx(domain=mesh))

def fitfunc(x, a, b):
    return a * (x/b - 1)**-0.5

def fitfunc2(x, a, b):
    return a * (1 + (x/b-1)**-1) * (x/b - 1)**-0.5

def mesh_to_boundary(v, b_mesh):
    """
    Returns a the boundary representation of the CG-1 function v
    """
    # Extract the underlying volume and boundary meshes
    mesh = v.function_space().mesh()

    # We use a Dof->Vertex mapping to create a global
    # array with all DOF values ordered by mesh vertices
    DofToVert = df.dof_to_vertex_map(v.function_space())
    VGlobal = np.zeros(v.vector().size())

    vec = v.vector().get_local()
    for i in range(len(vec)):
        Vert = df.MeshEntity(mesh, 0, DofToVert[i])
        globalIndex = Vert.global_index()
        VGlobal[globalIndex] = vec[i]
    VGlobal = SyncSum(VGlobal)

    # Use the inverse mapping to se the DOF values of a boundary
    # function
    surface_space = df.FunctionSpace(b_mesh, "CG", 1)
    surface_function = df.Function(surface_space)
    mapa = b_mesh.entity_map(0)
    DofToVert = df.dof_to_vertex_map(df.FunctionSpace(b_mesh, "CG", 1))

    LocValues = surface_function.vector().get_local()
    for i in range(len(LocValues)):
        VolVert = df.MeshEntity(mesh, 0, mapa[int(DofToVert[i])])
        GlobalIndex = VolVert.global_index()
        LocValues[i] = VGlobal[GlobalIndex]

    surface_function.vector().set_local(LocValues)
    surface_function.vector().apply('')
    return surface_function

def vector_mesh_to_boundary(func, b_mesh):
    v_split = func.split(deepcopy=True)
    v_b = []
    for v in v_split:
        v_b.append(mesh_to_boundary(v, b_mesh))
    Vb = df.VectorFunctionSpace(b_mesh, "CG", 1)
    vb_out = df.Function(Vb)
    scalar_to_vec = df.FunctionAssigner(Vb, [v.function_space() for
                                                  v in v_b])
    scalar_to_vec.assign(vb_out, v_b)
    return vb_out


def SyncSum(vec):
    """ Returns sum of vec over all mpi processes.
    Each vec vector must have the same dimension for each MPI process """

    comm = df.MPI.comm_world
    NormalsAllProcs = np.zeros(comm.Get_size() * len(vec), dtype=vec.dtype)
    comm.Allgather(vec, NormalsAllProcs)

    out = np.zeros(len(vec))
    for j in range(comm.Get_size()):
        out += NormalsAllProcs[len(vec) * j:len(vec) * (j + 1)]
    return out

def boundary_to_mesh(f, mesh):
    b_mesh = f.function_space().mesh()
    SpaceV = df.FunctionSpace(mesh, "CG", 1)
    SpaceB = df.FunctionSpace(b_mesh, "CG", 1)

    F = df.Function(SpaceV)
    GValues = np.zeros(F.vector().size())

    map = b_mesh.entity_map(0)  # Vertex map from boundary mesh to parent mesh
    d2v = df.dof_to_vertex_map(SpaceB)
    v2d = df.vertex_to_dof_map(SpaceV)

    dof = SpaceV.dofmap()
    imin, imax = dof.ownership_range()

    for i in range(f.vector().local_size()):
        GVertID = df.Vertex(b_mesh, d2v[i]).index()  # Local Vertex ID for given dof on boundary mesh
        PVertID = map[GVertID]  # Local Vertex ID of parent mesh
        PDof = v2d[PVertID]  # Dof on parent mesh
        value = f.vector()[i]  # Value on local processor
        GValues[dof.local_to_global_index(PDof)] = value
    GValues = SyncSum(GValues)

    F.vector().set_local(GValues[imin:imax])
    F.vector().apply("")
    return F


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

    # Process each mesh in parallel
    for fname in fnames:
        print(fname)
        prm = dict([key_val_split(item) for item in fname[:-4].split("_")])

        #mesh_fname = os.path.join(meshfolder, fname + ".h5")
        subd_fname = os.path.join(datafolder, fname + "_subd.xdmf")
        output_fname = os.path.join(datafolder, fname + "_geom.dat")

        """
        mesh = df.Mesh()
        with df.HDF5File(mesh.mpi_comm(), mesh_fname, "r") as h5f:
            h5f.read(mesh, "mesh", False)

        n = df.FacetNormal(mesh)
        V = df.VectorFunctionSpace(mesh, "CG", 1)
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        a = df.inner(u,v)*df.ds
        l = df.inner(n, v)*df.ds
        A = df.assemble(a, keep_diagonal=True)
        L = df.assemble(l)

        A.ident_zeros()
        nh = df.Function(V)

        df.solve(A, nh.vector(), L)
        df.File(os.path.join(datafolder, fname + "_nh.pvd")) << nh
        bmesh = df.BoundaryMesh(mesh, "exterior")
        nb = vector_mesh_to_boundary(nh, bmesh)
        Q = df.FunctionSpace(bmesh, "CG", 1)

        p, q = df.TrialFunction(Q), df.TestFunction(Q)
        a = df.inner(p,q)*df.dx
        l = df.inner(df.div(nb), q)*df.dx
        A = df.assemble(a, keep_diagonal=True)
        L = df.assemble(l)
        A.ident_zeros()
        kappab = df.Function(Q)
        df.solve(A, kappab.vector(), L)
        kappa = boundary_to_mesh(kappab, mesh)

        print(df.assemble(kappa*df.ds))
        df.File(os.path.join(datafolder, fname + "_kappa.pvd")) << kappa
        """
        
        m = meshio.read(subd_fname)

        nodes = m.points
        faces = m.cells_dict["triangle"]
        subd = m.cell_data["subd"][0].flatten()

        # Inlet
        nodes_inlet, faces_inlet = get_submesh(nodes, faces, subd == 3)
        nodes_outlet, faces_outlet = get_submesh(nodes, faces, subd == 4)
        nodes_bdry, faces_bdry = get_submesh(nodes, faces, subd == 7)

        inlet_mesh_fname = os.path.join(datafolder, fname + "_subd3.h5")
        numpy_to_dolfin_file(nodes_inlet[:, [1, 2]], faces_inlet, filename=inlet_mesh_fname)

        outlet_mesh_fname = os.path.join(datafolder, fname + "_subd4.h5")
        numpy_to_dolfin_file(nodes_outlet[:, [1, 2]], faces_outlet, filename=outlet_mesh_fname)

        #bdry_mesh_fname = os.path.join(datafolder, fname + "_subd7.h5")
        #numpy_to_dolfin_file(nodes_bdry[:, [0, 2]], faces_bdry, filename=bdry_mesh_fname)

        inlet_area = compute_mesh_area(inlet_mesh_fname)
        outlet_area = compute_mesh_area(outlet_mesh_fname)

        gaussian_curv = igl.gaussian_curvature(nodes_bdry, faces_bdry)

        l = igl.cotmatrix(nodes_bdry, faces_bdry)
        m = igl.massmatrix(nodes_bdry, faces_bdry, igl.MASSMATRIX_TYPE_VORONOI)

        minv = sp.sparse.diags(1 / m.diagonal())

        hn = -minv.dot(l.dot(nodes_bdry))
        mean_curv = np.linalg.norm(hn, axis=1)

        bdry_mesh_fname = os.path.join(datafolder, fname + "_subd7.xdmf")

        #tm = trimesh.Trimesh(nodes_bdry, faces_bdry)

        meshio.write_points_cells(bdry_mesh_fname, nodes_bdry, cells=[("triangle", faces_bdry)], point_data=dict(gaussian_curv=gaussian_curv, mean_curv=mean_curv))

        #fig, ax = plt.subplots(1, 2)
        #ax[0].hist(mean_curv[gaussian_curv < 1], bins=100)
        #ax[1].hist(gaussian_curv, bins=100)
        #plt.show()

        H = mean_curv[gaussian_curv < 1].mean()

        Pc = prm["Pc"][0]
        dist = prm["Dist"][0]

        data_loc = [dist, Pc, inlet_area, outlet_area, H]
        print(*data_loc)
        data_.append(data_loc)
        np.savetxt(output_fname, np.array(data_loc))

    data_ = np.array(data_)

    if args.show:
        fig, ax = plt.subplots(1, 5)

        a_ = []
        b_ = []

        dists = sorted(np.unique(data_[:, 0]))
        for dist in dists:
            data_loc = data_[data_[:, 0] == dist, :]

            Pc = data_loc[:, 1]

            ax[0].plot(Pc, data_loc[:, 2], '*', label=f"$d={dist}$")
            ax[1].plot(Pc, data_loc[:, 3], '*')
            popt = np.polyfit(Pc, data_loc[:, 3], 1)
            ax[1].plot(Pc, popt[1] + popt[0] * Pc, 'k--')
            a_.append(popt[1])
            b_.append(popt[0])

            ax[2].plot(Pc, data_loc[:, 4], '*')
        
        popt, pcov = opt.curve_fit(fitfunc, data_[:, 1], data_[:, 2])
        popt2, pcov2 = opt.curve_fit(fitfunc2, data_[:, 1], data_[:, 2])

        pc_ = np.linspace(data_[:, 1].min(), data_[:, 1].max(), 1000)

        popt3 = np.polyfit(data_[:, 1], data_[:, 4], 1)
        ax[2].plot(pc_, pc_*popt3[0], 'k--')

        ax[0].plot(pc_, fitfunc(pc_, *popt))
        ax[0].plot(pc_, fitfunc2(pc_, *popt2))

        ax[3].plot(dists, np.array(a_))
        ax[4].plot(dists, np.array(b_))

        plt.show()