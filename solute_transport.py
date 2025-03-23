import argparse
import numpy as np
import dolfin as df
from utils import mpi_max, mpi_min, mpi_print, Top, Btm, Boundary, Sphere, mpi_is_root
import os
from utils import numpy_to_dolfin_file, dolfin_file_to_numpy, double_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Solve for permeability in a single geometry")
    #parser.add_argument("filename", type=str, help="Name of the .stl file")
    parser.add_argument("meshfolder", type=str, help="Name of folder containing .h5 volume mesh files")
    parser.add_argument("flowfolder", type=str, help="Name of folder containing .h5 flow fields")
    parser.add_argument("outfolder", type=str, help="Output folder")
    return parser.parse_args()

def get_files(folder, ext=".h5"):
    files = []
    for item in os.listdir(folder):
        if item[-len(ext):] == ext:
            files.append(item)
    return files

class PBCx(df.SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left boundary AND NOT on the two slave edge
        return bool(df.near(x[0], 0.) and not df.near(x[0], self.Lx) and on_boundary)

    def map(self, x, y):
        if df.near(x[0], self.Lx):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] #- self.Ly
            y[2] = x[2]


if __name__ == "__main__":
    args = parse_args()

    velocity_degree = 2
    pressure_degree = 1

    mesh_files = set(get_files(args.meshfolder))
    flow_files = set(get_files(args.flowfolder))
    files = sorted(list(mesh_files & flow_files))

    if mpi_is_root and not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    for filename in files[:]:
        folder = os.path.join(args.outfolder, filename[:-3])
        if mpi_is_root and not os.path.exists(folder):
            os.makedirs(folder)

        #mesh2name = os.path.join(args.outfolder, filename[:-3] + "_doubled_mesh.h5")
        #flow2name = os.path.join(args.outfolder, filename[:-3] + "_doubled_flow.h5")
        flow2name = os.path.join(folder, "flow.h5")

        make_new_flow = os.path.exists(flow2name)

        mesh2 = df.Mesh()
        if make_new_flow:
            with df.HDF5File(mesh2.mpi_comm(), flow2name, "r") as h5f:
                h5f.read(mesh2, "mesh", False)

            Lx = mpi_max(mesh2.coordinates()[:, 0])
            pbcx = PBCx(Lx)

            V2 = df.VectorFunctionSpace(mesh2, "CG", velocity_degree, constrained_domain=pbcx)
            u2_ = df.Function(V2, name="u")

            with df.HDF5File(mesh2.mpi_comm(), flow2name, "r") as h5f:
                h5f.read(u2_, "u")
        else:
            mesh = df.Mesh()
            with df.HDF5File(mesh.mpi_comm(), os.path.join(args.meshfolder, filename), "r") as h5f:
                h5f.read(mesh, "mesh", False)
            x = mesh.coordinates()[:]
            x_max = mpi_max(x)
            d = x_max[2]
            x[:, :] /= d
            mesh.coordinates()[:] = x
            
            Lx = 2*mpi_max(mesh.coordinates()[:, 0])

            V = df.VectorFunctionSpace(mesh, "CG", 2)
            u_ = df.Function(V, name="u")
            with df.HDF5File(mesh.mpi_comm(), os.path.join(args.flowfolder, filename), "r") as h5f:
                h5f.read(u_, "u")

            if mpi_is_root:
                node, elem = dolfin_file_to_numpy(os.path.join(args.meshfolder, filename))
                node, elem = double_mesh(node/d, elem, axis="x", reflect=True)
                numpy_to_dolfin_file(node, elem, flow2name)

            with df.HDF5File(mesh2.mpi_comm(), flow2name, "r") as h5f:
                h5f.read(mesh2, "mesh", False)

            pbcx = PBCx(Lx)

            V2 = df.VectorFunctionSpace(mesh2, "CG", velocity_degree, constrained_domain=pbcx)
            u2_ = df.Function(V2, name="u")
            df.LagrangeInterpolator.interpolate(u2_, u_)

            meshflip = df.Mesh(mesh)
            meshflip.coordinates()[:, 0] *= -1
            meshflip.coordinates()[:, 0] += Lx

            Vflip = df.VectorFunctionSpace(meshflip, "CG", 2)
            uflip_ = df.Function(Vflip, name="u")
            uu = u_.vector()[:]
            uu[1::3] *= -1
            uu[2::3] *= -1
            uflip_.vector()[:] = uu

            u2flip_ = df.Function(V2, name="u")
            df.LagrangeInterpolator.interpolate(u2flip_, uflip_)

            anyval = np.asarray(np.logical_and(u2_.vector()[:] != 0., u2flip_.vector()[:] != 0.), dtype=int)

            u2_.vector()[:] += u2flip_.vector()[:]
            u2_.vector()[:] /= (anyval+1)

            S = df.FunctionSpace(mesh2, "CG", 1, constrained_domain=pbcx)
            
            V_Omega = df.assemble(df.Constant(1.)*df.dx(domain=mesh2))
            u2x_mean = df.assemble(u2_[0] * df.dx) / V_Omega
            u2_.vector()[:] /= u2x_mean
            p2_ = df.Function(S, name="p")

            with df.HDF5File(mesh2.mpi_comm(), flow2name, "w") as h5f:
                h5f.write(mesh2, "mesh")
                h5f.write(u2_, "u")
                h5f.write(p2_, "p")
        
            with df.XDMFFile(mesh2.mpi_comm(), os.path.join(folder, "u_show.xdmf")) as xdmff:
                xdmff.write(u2_)

            if mpi_is_root:
                with open(os.path.join(folder, "dolfin_params.dat"), "w") as ofile:
                    ofile.write("velocity_space=P{}\n".format(velocity_degree))
                    ofile.write("pressure_space=P{}\n".format(pressure_degree))
                    ofile.write("timestamps=timestamps.dat\n")
                    ofile.write("mesh=flow.h5\n")
                    ofile.write("periodic_x=true\n")
                    ofile.write("periodic_y=false\n")
                    ofile.write("periodic_z=false\n")
                    ofile.write("rho=1.0\n")
                    ofile.write("ignore_pressure=true")

                with open(os.path.join(folder, "timestamps.dat"), "w") as ofile:
                    t_ = [0., 1e9]
                    for tstep, t in enumerate(t_):
                        ofile.write("{} {}\n".format(t, "flow.h5"))

        S = df.FunctionSpace(mesh2, "CG", 1, constrained_domain=pbcx)
        tol = 1e-7
        
        x = mesh2.coordinates()[:]
        x_max = mpi_max(x)
        x_min = mpi_min(x)

        bdry = Boundary()
        outlet = Top(x_min, x_max, tol=tol, direction=0)
        inlet = Btm(x_min, x_max, tol=tol, direction=0)
        
        # Boundaries
        subd = df.MeshFunction("size_t", mesh2, mesh2.topology().dim() - 1)
        subd.rename("subd", "subd")
        subd.set_all(0)
        bdry.mark(subd, 1)
        inlet.mark(subd, 2)
        outlet.mark(subd, 2)

        n = df.FacetNormal(mesh2)

        Pe = df.Constant(1)

        chi = df.TrialFunction(S)
        chi_ = df.Function(S, name="chi")
        psi = df.TestFunction(S)

        ds = df.Measure("ds", domain=mesh2, subdomain_data=subd)

        one = df.interpolate(df.Constant(1.), S)
        V_Omega = df.assemble(one*df.dx)
        u2x_mean = df.assemble(u2_[0] * df.dx) / V_Omega
        u2_.vector()[:] /= u2x_mean

        F_chi = (n[0]*psi*ds(1)
                + df.inner(df.grad(chi), df.grad(psi))*df.dx
                + Pe*psi*df.dot(u2_, df.grad(chi))*df.dx
                + Pe*(u2_[0] - df.Constant(1.))*psi*df.dx)
        
        a_chi, L_chi = df.lhs(F_chi), df.rhs(F_chi)

        problem_chi2 = df.LinearVariationalProblem(a_chi, L_chi, chi_, bcs=[])
        solver_chi2 = df.LinearVariationalSolver(problem_chi2)

        #solver_chi2.parameters["linear_solver"] = "gmres"
        solver_chi2.parameters["krylov_solver"]["absolute_tolerance"] = 1e-15

        Pe_val_min = -1
        Pe_val_max = 3
        Pe_vals = np.logspace(Pe_val_min, Pe_val_max, (Pe_val_max-Pe_val_min)*4+1)

        xdmff = df.XDMFFile(mesh2.mpi_comm(), os.path.join(folder, "chi.xdmf"))
        xdmff.parameters["rewrite_function_mesh"] = False
        xdmff.parameters["functions_share_mesh"] = True
        xdmff.parameters["flush_output"] = True

        data_ = []
        for Pe_val in Pe_vals:
            Pe.assign(Pe_val)
            solver_chi2.solve()
            chi_mean = df.assemble(chi_ * df.dx) / V_Omega
            chi_.vector()[:] -= chi_mean

            integrals = [1, 2*df.assemble(chi_.dx(0)*df.dx)/V_Omega,
                        df.assemble(df.inner(df.grad(chi_), df.grad(chi_))*df.dx)/V_Omega]

            mpi_print(Pe_val, integrals)
            if mpi_is_root:
                data_.append([Pe_val, sum(integrals)])

            xdmff.write(chi_, Pe_val)

        if mpi_is_root:
            data_ = np.array(data_)
            np.savetxt(os.path.join(folder, "Bsims.dat"), data_)