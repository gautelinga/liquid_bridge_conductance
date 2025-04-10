import argparse
import numpy as np
import dolfin as df
from utils import mpi_max, mpi_min, mpi_print, Top, Btm, Boundary, Sphere, mpi_is_root
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Solve for permeability in a single geometry")
    #parser.add_argument("filename", type=str, help="Name of the .stl file")
    parser.add_argument("infolder", type=str, help="Name of folder containing .h5 volume mesh files")
    parser.add_argument("outfolder", type=str, help="Name of output folder")
    parser.add_argument("--beta", type=float, default=100, help="Parameter to weakly enforce boundary condition")
    parser.add_argument("--noslip", action="store_true", help="Enforce no-slip condition at the air-water boundary")
    parser.add_argument("--direct", action="store_true", help="Use direct solver")
    parser.add_argument("--regenerate", action="store_true", help="")
    return parser.parse_args()

def simulate(infilename, outfilename, beta, use_noslip=False, use_direct=True, tol = 1e-7):
    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), infilename, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    # mesh = df.refine(mesh)

    x = mesh.coordinates()[:]
    x_max = mpi_max(x)
    x_min = mpi_min(x)
    mpi_print("Original dimensions:", x_min, x_max)

    D = x_max[2]
    x[:, :] /= D
    mesh.coordinates()[:] = x

    x_max = mpi_max(x)
    x_min = mpi_min(x)
    mpi_print("Scaled dimensions:", x_min, x_max)

    bdry = Boundary()
    top = Top(x_min, x_max, tol=tol, direction=2)
    btm = Btm(x_min, x_max, tol=tol, direction=2)
    outlet = Top(x_min, x_max, tol=tol, direction=0)
    inlet = Btm(x_min, x_max, tol=tol, direction=0)
    inside = Btm(x_min, x_max, tol=tol, direction=1)
    sphere = Sphere(x_min, x_max, tol=tol, direction=2)

    # Boundaries
    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.rename("subd", "subd")
    subd.set_all(0)
    bdry.mark(subd, 7)
    sphere.mark(subd, 6)
    top.mark(subd, 1)
    btm.mark(subd, 2)
    inlet.mark(subd, 3)
    outlet.mark(subd, 4)
    inside.mark(subd, 5)

    with df.XDMFFile(mesh.mpi_comm(), outfilename + "_subd.xdmf") as xdmff:
        xdmff.write(subd)

    u_el = df.VectorElement("CG", mesh.ufl_cell(), 2)
    p_el = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    W = df.FunctionSpace(mesh, df.MixedElement([u_el, p_el]))

    u, p = df.TrialFunctions(W)
    v, q = df.TestFunctions(W)

    f = df.Constant((1., 0., 0.))

    ds = df.Measure("ds", subdomain_data=subd)
    n = df.FacetNormal(mesh)
    delta = mesh.hmin()
    beta = df.Constant(beta)

    symgrad = lambda u: df.sym(df.grad(u))
    Sigma = lambda u, p: 2*symgrad(u) - p*df.Identity(3)
    t = lambda u, p: df.dot(Sigma(u, p), n)

    #a = df.inner(df.grad(u), df.grad(v)) * df.dx - p * df.div(v) * df.dx - q * df.div(u) * df.dx
    a = 2*df.inner(symgrad(u), symgrad(v)) * df.dx - p * df.div(v) * df.dx - q * df.div(u) * df.dx
    L = df.dot(f, v) * df.dx

    # preconditioning matrix
    #m = df.inner(df.grad(u), df.grad(v)) * df.dx + p * q * df.dx
    m = 2*df.inner(symgrad(u), symgrad(v)) * df.dx + p * q * df.dx

    noslip = df.Constant((0., 0., 0.))
    
    bc_inlet = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 3)
    bc_outlet = df.DirichletBC(W.sub(1), df.Constant(0.), subd, 4)
    bc_top = df.DirichletBC(W.sub(0), noslip, subd, 1)
    bc_btm = df.DirichletBC(W.sub(0), noslip, subd, 2)
    bc_sphere = df.DirichletBC(W.sub(0), noslip, subd, 6)
    bc_bdry = df.DirichletBC(W.sub(0), noslip, subd, 7)  # change
    bc_inside = df.DirichletBC(W.sub(0).sub(1), df.Constant(0.), subd, 5)

    bc_uy_inlet = df.DirichletBC(W.sub(0).sub(1), df.Constant(0.), subd, 3)
    bc_uz_inlet = df.DirichletBC(W.sub(0).sub(2), df.Constant(0.), subd, 3)
    bc_uy_outlet = df.DirichletBC(W.sub(0).sub(1), df.Constant(0.), subd, 4)
    bc_uz_outlet = df.DirichletBC(W.sub(0).sub(2), df.Constant(0.), subd, 4)

    bcs = [bc_inlet, bc_outlet, bc_top, bc_btm, bc_sphere, bc_inside, bc_uy_inlet, bc_uz_inlet, bc_uy_outlet, bc_uz_outlet]

    if use_noslip:
        bcs.append(bc_bdry)
    else:
        a += 2*df.inner(df.dot(symgrad(u), n), v)*ds(3) + 2*df.inner(df.dot(symgrad(u), n), v)*ds(4)
        m += 2*df.inner(df.dot(symgrad(u), n), v)*ds(3) + 2*df.inner(df.dot(symgrad(u), n), v)*ds(4)

        # Weakly include free-slip conditions
        a += - df.inner(df.dot(t(u, p), n), df.dot(v, n)) * ds(7) \
            - df.inner(df.dot(u, n), df.dot(t(v, q), n)) * ds(7) \
            + beta*delta**-1*df.inner(df.dot(u, n), df.dot(v, n))*ds(7)
        
        m += beta*delta**-1*df.inner(df.dot(u, n), df.dot(v, n))*ds(7)

    w_ = df.Function(W)

    if use_direct:
        df.solve(a == L, w_, bcs)
    else:
        A, b = df.assemble_system(a, L, bcs)

        P, _ = df.assemble_system(m, L, bcs)

        A_ = df.as_backend_type(A).mat()
        b_ = df.as_backend_type(b).vec()
        P_ = df.as_backend_type(P).mat()

        rtol = 1e-9

        from petsc4py import PETSc
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setOperators(A_, P_)
        ksp.setType("minres")
        ksp.setTolerances(rtol=1e-9)
        ksp.getPC().setType("fieldsplit")
        ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

        pc = ksp.getPC()
        is0 = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
        is1 = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
        fields = [('u', is0), ('p', is1)]
        pc.setFieldSplitIS(*fields)

        # Set the preconditioners for each block
        ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        ksp_u.getPC().setType("gamg")
        ksp_p.setType("preonly")
        ksp_p.getPC().setType("jacobi")


        df.PETScOptions.set('ksp_view')
        df.PETScOptions.set('ksp_monitor_true_residual')
        # df.PETScOptions.set('ksp_rtol', rtol)

        # df.PETScOptions.set('ksp_type', 'minres')
        # df.PETScOptions.set('pc_type', 'fieldsplit')
        # df.PETScOptions.set('pc_fieldsplit_detect_saddle_point')

        # df.PETScOptions.set('pc_fieldsplit_type', 'schur') # 'additive')
        # df.PETScOptions.set('pc_fieldsplit_schur_fact_type', 'diag')
        # df.PETScOptions.set('pc_fieldsplit_schur_precondition', 'user')

        # df.PETScOptions.set('fieldsplit_u_ksp_type', 'preonly')
        # df.PETScOptions.set('fieldsplit_u_pc_type', 'gamg') #'lu')
        # df.PETScOptions.set('fieldsplit_p_ksp_type', 'preonly')
        # df.PETScOptions.set('fieldsplit_p_pc_type', 'jacobi')

        ksp.setFromOptions()

        x_, _ = A_.createVecs()

        ksp.solve(b_, x_)
        w_.vector()[:] = x_

        for bc in bcs:
            bc.apply(w_.vector())

    u_, p_ = w_.split(deepcopy=True)
    u_.rename("u", "u")
    p_.rename("p", "p")

    x_ = df.interpolate(df.Expression("x[0]", degree=1), W.sub(1).collapse())
    p_.vector()[:] += -x_.vector()[:]

    volume = df.assemble(df.Constant(1.0) * df.dx(domain=mesh))
    bbdist = 2*x_max[0]
    flux_inlet = df.assemble(u_[0] * ds(3))
    flux_outlet = df.assemble(u_[0] * ds(4))
    flux_avg = df.assemble(u_[0] * df.dx) / x_max[0]
    flux_bdry = df.assemble(df.dot(n, u_) * ds(7))
    mpi_print(f"volume = {volume}, bead-bead distance = {bbdist}, flux_in={flux_inlet}, flux_out={flux_outlet}, flux_avg={flux_avg}, flux_bdry={flux_bdry}")

    k0 = flux_avg

    volume_phys = volume * D**3
    bbdist_phys = bbdist * D
    k_phys = k0 * D**2

    mpi_print(bbdist_phys, volume_phys, k_phys)

    with df.HDF5File(mesh.mpi_comm(), outfilename + ".h5", "w") as h5f:
        h5f.write(u_, "u")
        h5f.write(p_, "p")

    with df.XDMFFile(mesh.mpi_comm(), outfilename + "_fields.xdmf") as xdmff:
        xdmff.parameters["flush_output"] = True
        xdmff.parameters["functions_share_mesh"] = True
        xdmff.write(u_, 0.)
        xdmff.write(p_, 0.)

    data = [volume, bbdist, flux_inlet, flux_outlet, flux_avg, flux_bdry, k0, D, volume_phys, bbdist_phys, k_phys]

    return data


if __name__ == "__main__":
    args = parse_args()

    fnames = []
    for fname in os.listdir(args.infolder):
        if fname[-3:] == ".h5":
            fnames.append(fname)
    fnames = sorted(fnames)

    for fname in fnames:
        infilename = os.path.join(args.infolder, fname)
        outfilename = os.path.join(args.outfolder, fname[:-3])
        
        if not os.path.exists(outfilename + ".dat") or args.regenerate:
            mpi_print(fname)
                
            data = simulate(infilename, outfilename, args.beta, args.noslip, args.direct, tol=1e-4)

            if mpi_is_root:
                np.savetxt(outfilename + ".dat", np.array(data))