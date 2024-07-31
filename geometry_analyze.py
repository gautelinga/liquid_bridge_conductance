import os
from utils import mpi_is_root, mpi_rank, mpi_size, numpy_to_dolfin_file, key_val_split
import argparse
import dolfin as df
import meshio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
import trimesh
import pymeshlab

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

    if False:
        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(nodes_sub, faces_sub)
        ms.add_mesh(m)

        #print(ms.mesh_number())

        #ms.apply_filter("meshing_re_orient_faces_coherently")
        ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=100)

        #print(ms.mesh_number())

        # get a reference to the current mesh
        m = ms.current_mesh()

        # get numpy arrays of vertices and faces of the current mesh
        nodes_sub = m.vertex_matrix()
        faces_sub = m.face_matrix()

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

def get_curv(nodes_bdry, faces_bdry):
    gaussian_curv = igl.gaussian_curvature(nodes_bdry, faces_bdry)

    l = igl.cotmatrix(nodes_bdry, faces_bdry)
    m = igl.massmatrix(nodes_bdry, faces_bdry, igl.MASSMATRIX_TYPE_VORONOI)

    minv = sp.sparse.diags(1 / m.diagonal())

    hn = -minv.dot(l.dot(nodes_bdry))
    mean_curv = np.linalg.norm(hn, axis=1)

    return mean_curv, gaussian_curv

def get_curv(v, f):
    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    gaussian_curv = k1*k2
    mean_curv = 0.5*(k1 + k2)
    return mean_curv, gaussian_curv

def reorient_faces(nodes, faces):
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(nodes, faces)
    ms.add_mesh(m)

    #print(ms.mesh_number())

    ms.apply_filter("meshing_re_orient_faces_coherently")
    #print(ms)

    #print(ms.mesh_number())

    # get a reference to the current mesh
    m = ms.current_mesh()

    # get numpy arrays of vertices and faces of the current mesh
    nodes_out = m.vertex_matrix()
    faces_out = m.face_matrix()

    #print(faces_out.shape, faces.shape)
    assert(abs(nodes_out-nodes).max() < 1e-12)

    return nodes_out, faces_out

def get_curv2(nodes, faces):
    # Create a MeshSet object
    ms = pymeshlab.MeshSet()
    
    # Create a mesh from the nodes and faces arrays and add it to the MeshSet
    m = pymeshlab.Mesh(nodes, faces)
    ms.add_mesh(m)
    
    pymeshlab.print_pymeshlab_version()
    filters = pymeshlab.filter_list()
    for filter in filters:
        if "curvature" in filter:
            print(filter)

    # Apply the filter to compute curvature
    #ms.apply_filter("colorize_curvature_apss")
    ms.apply_filter("meshing_re_orient_faces_coherently")
    ms.apply_filter("compute_curvature_and_color_apss_per_vertex")
    curvature_mesh = ms.current_mesh()

    ms.save_current_mesh("test.ply")

    #print(curvature_mesh.__dict__)

    # Get the mean curvature per vertex
    v_curv = curvature_mesh.vertex_scalar_field_values('MeanCurvature')
    return v_curv

def get_boundary_points(v, f):
    e = dict()
    for iface, vloc_ in enumerate(f):
        v1, v2, v3 = list(sorted(vloc_))
        edges = [(v1, v2), (v1, v3), (v2, v3)]
        for edge in edges:
            if edge in e:
                e[edge].append(iface)
            else:
                e[edge] = [iface]

    boundary_nodes = set()
    for key, val in e.items():
        if len(val) == 1:
            boundary_nodes.update(set(key))

    return list(sorted(boundary_nodes))

def get_voronoi_area(v, f):
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI).diagonal()
    return m

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
        
        m = meshio.read(subd_fname)

        nodes = m.points
        faces = m.cells_dict["triangle"]
        subd = m.cell_data["subd"][0].flatten()

        exterior_faces = subd > 0
        faces = faces[exterior_faces, :]
        subd = subd[exterior_faces]

        #print(subd)
        #mean_curv, gaussian_curv = get_curv(nodes, faces)
        nodes, faces = reorient_faces(nodes, faces)

        # Inlet
        nodes_inlet, faces_inlet = get_submesh(nodes, faces, subd == 3)
        nodes_outlet, faces_outlet = get_submesh(nodes, faces, subd == 4)
        nodes_bdry, faces_bdry = get_submesh(nodes, faces, subd == 7)
        nodes_bead, faces_bead = get_submesh(nodes, faces, subd == 6)

        # Optimal radii
        dx = np.copy(nodes_bead[:, :])
        dx[:, 2] -= 0.5
        dr = np.linalg.norm(dx, axis=1)

        if False:
            n = dx
            for dim in range(3):
                n[:, dim] /= dr
            nodes_bead = n*0.5

        inlet_mesh_fname = os.path.join(datafolder, fname + "_subd3.h5")
        numpy_to_dolfin_file(nodes_inlet[:, [1, 2]], faces_inlet, filename=inlet_mesh_fname)

        outlet_mesh_fname = os.path.join(datafolder, fname + "_subd4.h5")
        numpy_to_dolfin_file(nodes_outlet[:, [1, 2]], faces_outlet, filename=outlet_mesh_fname)

        #bead_mesh_fname = os.path.join(datafolder, fname + "_subd6.h5")

        #bdry_mesh_fname = os.path.join(datafolder, fname + "_subd7.h5")
        #numpy_to_dolfin_file(nodes_bdry[:, [0, 2]], faces_bdry, filename=bdry_mesh_fname)

        inlet_area = compute_mesh_area(inlet_mesh_fname)
        outlet_area = compute_mesh_area(outlet_mesh_fname)

        mean_curv_bdry, gaussian_curv_bdry = get_curv(nodes_bdry, faces_bdry)
        bdry_mesh_fname = os.path.join(datafolder, fname + "_subd7.xdmf")
        mean_curv_bead, gaussian_curv_bead = get_curv(nodes_bead, faces_bead)
        bead_mesh_fname = os.path.join(datafolder, fname + "_subd6.xdmf")
        #bead_mesh_fname2 = os.path.join(datafolder, fname + "_subd6.ply")
        
        #tm = trimesh.Trimesh(nodes_bdry, faces_bdry)

        #fig, ax = plt.subplots(1, 2)
        #ax[0].hist(mean_curv[gaussian_curv < 1], bins=100)
        #ax[1].hist(gaussian_curv, bins=100)
        #plt.show()

        voronoi_area_bead = get_voronoi_area(nodes_bead, faces_bead)
        voronoi_area_bdry = get_voronoi_area(nodes_bdry, faces_bdry)

        boundary_nodes_bdry = get_boundary_points(nodes_bdry, faces_bdry)
        is_interior_bdry = np.ones_like(gaussian_curv_bdry, dtype=bool)
        is_interior_bdry[boundary_nodes_bdry] = False

        boundary_nodes_bead = get_boundary_points(nodes_bead, faces_bead)
        is_interior_bead = np.ones_like(gaussian_curv_bead, dtype=bool)
        is_interior_bead[boundary_nodes_bead] = False

        xx = nodes_bdry[boundary_nodes_bdry, :]
        ids = xx[:, 0] < 1e-3
        ids[xx[:, 2] > 0.5] = False

        xx = xx[ids, :]

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((xx[:, 1]-xc)**2 + (xx[:, 2]-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_guess = xx[:, 1].mean(), xx[:, 2].mean()
        center, ier = opt.leastsq(f_2, center_guess)
        radius = calc_R(*center).mean()

        if False:
            fig, ax = plt.subplots(1, 1)
            ax.scatter(xx[:, 1], xx[:, 2])
            circ = plt.Circle(center, radius, color='r', fill=False)
            ax.add_patch(circ)
            ax.set_aspect("equal")
            plt.show()

        meshio.write_points_cells(bdry_mesh_fname, nodes_bdry, cells=[("triangle", faces_bdry)],
                                  point_data=dict(gaussian_curv=gaussian_curv_bdry, mean_curv=mean_curv_bdry, 
                                                  area=voronoi_area_bdry, interior=np.asarray(is_interior_bdry, dtype=float)))

        meshio.write_points_cells(bead_mesh_fname, nodes_bead, cells=[("triangle", faces_bead)],
                                  point_data=dict(gaussian_curv=gaussian_curv_bead, mean_curv=mean_curv_bead,
                                                  area=voronoi_area_bead, dr=dr, interior=np.asarray(is_interior_bead, dtype=float)))

        #meshio.write_points_cells(bead_mesh_fname2, nodes_bead, cells=[("triangle", faces_bead)],
        #                         point_data=dict(gaussian_curv=gaussian_curv_bead, mean_curv=mean_curv_bead,
        #                                          area=voronoi_area_bead, interior=np.asarray(is_interior_bead, dtype=float)))

        K_mean_bdry = (gaussian_curv_bdry*voronoi_area_bdry)[is_interior_bdry].sum()/voronoi_area_bdry[is_interior_bdry].sum()
        K_mean_bead = (gaussian_curv_bead*voronoi_area_bead)[is_interior_bead].sum()/voronoi_area_bead[is_interior_bead].sum()

        H_mean_bdry = (mean_curv_bdry*voronoi_area_bdry)[is_interior_bdry].sum()/voronoi_area_bdry[is_interior_bdry].sum()
        H_mean_bead = (mean_curv_bead*voronoi_area_bead)[is_interior_bead].sum()/voronoi_area_bead[is_interior_bead].sum()

        Pc = prm["Pc"][0]
        dist = prm["Dist"][0]

        data_loc = [dist, Pc, inlet_area, outlet_area, H_mean_bdry, H_mean_bead, K_mean_bdry, K_mean_bead, radius, center[0], center[1]]
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