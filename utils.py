import h5py
import numpy as np
import dolfin as df
import os
import meshio
import mpi4py.MPI as MPI
mpi_comm = MPI.COMM_WORLD

# mpi_comm = MPI.comm_world
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_is_root = mpi_rank == 0

def mpi_barrier():
    MPI.barrier(mpi_comm)

def remove_safe(path):
    """ Remove file in a safe way. """
    if mpi_is_root and os.path.exists(path):
        os.remove(path)

def numpy_to_dolfin_file(nodes, elements, filename):
    """ Convert nodes and elements to a dolfin mesh file object. """
    dim = nodes.shape[1]
    npts = elements.shape[1]
    if dim == 0:
        celltype_str = "point"
    elif dim == 1:
        celltype_str = "interval"
    elif dim == 2:
        if npts == 3:
            celltype_str = "triangle"
        elif npts == 4:
            celltype_str = "quadrilateral"
    elif dim == 3:
        if npts == 4:
            celltype_str = "tetrahedron"
        elif npts == 8:
            celltype_str = "hexahedron"

    with h5py.File(filename, "w") as h5f:
        cell_indices = h5f.create_dataset(
            "mesh/cell_indices", data=np.arange(len(elements)),
            dtype='int64')
        topology = h5f.create_dataset(
            "mesh/topology", data=elements, dtype='int64')
        coordinates = h5f.create_dataset(
            "mesh/coordinates", data=nodes, dtype='float64')
        topology.attrs["celltype"] = np.string_(celltype_str)
        topology.attrs["partition"] = np.array([0], dtype='uint64')

def dolfin_file_to_numpy(mesh_file):
    with h5py.File(mesh_file, "r") as h5f:
        nodes = np.array(h5f["mesh/coordinates"])
        elements = np.array(h5f["mesh/topology"])
    return nodes, elements

def _sort_mesh(node, elem):
    sortids = np.argsort(node[:, 2])
    sortids_inv = np.zeros_like(sortids)
    for i, j in enumerate(sortids):
        sortids_inv[j] = i

    node_sorted = node[sortids, :]
    elem_sorted = sortids_inv[elem.flatten()].reshape(elem.shape)

    node = node_sorted
    elem = elem_sorted
    return node, elem

def double_mesh(node, elem, axis="z", reflect=False, tol=1e-7):
    if axis in ("x", 0):
        node_map = [1, 2, 0]
    elif axis in ("y", 1):
        node_map = [2, 0, 1]
    else:  # axis in ("z", 2):
        node_map = [0, 1, 2]
    node_map = list(zip(range(len(node_map)), node_map))

    node_cp = np.zeros_like(node)
    for i, j in node_map:
        node_cp[:, i] = node[:, j]
    node[:, :] = node_cp[:, :]

    node, elem = _sort_mesh(node, elem)

    x_max = node.max(0)
    x_min = node.min(0)

    if bool(np.sum(node[:, 2] == x_max[2]) !=
            np.sum(node[:, 2] > x_max[2]-tol)):
        node[node[:, 2] > x_max[2]-tol, 2] = x_max[2]

    node_new = np.zeros_like(node)
    elem_new = np.zeros_like(elem)
    node_new[:, :] = node[:, :]
    elem_new[:, :] = elem[:, :]
    if not reflect:
        node_new[:, 2] += x_max[2]-x_min[2]
    else:
        node_new[:, 2] = 2*x_max[2]-node_new[:, 2]
        node_new, elem_new = _sort_mesh(node_new, elem_new)

    glue_ids_old = np.argwhere(node[:, 2] == x_max[2]).flatten()
    glue_ids_new = np.argwhere(node_new[:, 2] == x_max[2]).flatten()

    x_old = node[glue_ids_old, :]
    x_new = node_new[glue_ids_new, :]

    sortids_y_old = np.lexsort((x_old[:, 1], x_old[:, 0]))
    sortids_y_new = np.lexsort((x_new[:, 1], x_new[:, 0]))

    glue_ids_old_out = glue_ids_old[sortids_y_old]
    glue_ids_new_out = glue_ids_new[sortids_y_new]

    for i_old, i_new in zip(glue_ids_old_out, glue_ids_new_out):
        assert not any(node[i_old, :] - node_new[i_new, :])

    ids_map = np.zeros(len(node), dtype=int)
    for i_old, i_new in zip(glue_ids_old_out, glue_ids_new_out):
        ids_map[i_new] = i_old

    ids_map[len(glue_ids_old_out):] = np.arange(
        len(node), 2*len(node)-len(glue_ids_old_out))

    elem_new = ids_map[elem_new.flatten()].reshape(elem_new.shape)

    node_out = np.vstack((node, node_new[len(glue_ids_old_out):, :]))
    elem_out = np.vstack((elem, elem_new))

    node_cp = np.zeros_like(node_out)
    for i, j in node_map:
        node_cp[:, j] = node_out[:, i]
    node_out[:, :] = node_cp[:, :]
    return node_out, elem_out

def mpi_print(*args):
    if mpi_rank == 0:
        print(*args)

def mpi_max(x):
    x_max_loc = x.max(axis=0)
    x_max = np.zeros_like(x_max_loc)
    mpi_comm.Allreduce(x_max_loc, x_max, op=MPI.MAX)
    return x_max

def mpi_min(x):
    x_min_loc = x.min(axis=0)
    x_min = np.zeros_like(x_min_loc)
    mpi_comm.Allreduce(x_min_loc, x_min, op=MPI.MIN)
    return x_min

def mpi_sum(data):
    data = mpi_comm.gather(data, root=0)
    if mpi_is_root:
        data = sum(data)
    else:
        data = 0    
    return data

class Boundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class GenSubDomain(df.SubDomain):
    def __init__(self, x_min, x_max, tol=df.DOLFIN_EPS_LARGE, direction=2):
        self.x_min = x_min
        self.x_max = x_max
        self.tol = tol
        self.direction = direction
        super().__init__()

class Top(GenSubDomain):
    def inside(self, x, on_boundary):
      return on_boundary and x[self.direction] > self.x_max[self.direction] - self.tol
    
class Btm(GenSubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[self.direction] < self.x_min[self.direction] + self.tol
    
class Sphere(GenSubDomain):
    def __init__(self, x_min, x_max, tol=df.DOLFIN_EPS_LARGE, direction=2):
        super().__init__(x_min, x_max, tol, direction)
        self.x_mid = np.copy(x_min)
        self.R = (x_max[direction] - x_min[direction]) / 2
        self.x_mid[direction] += self.R

    def inside(self, x, on_boundary):
        rad = np.linalg.norm(x - self.x_mid)
        return on_boundary and rad < self.R
    
def key_val_split(item, delim="="):
    key, val = item.split("=")

    return key, (float(val[:-2]), val[-2:])