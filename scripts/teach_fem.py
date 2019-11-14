# -*- coding: utf-8 -*-
# Python packages
import math
import numpy
import os
import scipy
import scipy.io as sio
import scipy.sparse.linalg
from matplotlib import gridspec, pylab as pl
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.preprocessing import normalize

""" FOR CLASS ROOM ************************************************************ """
import sys
sys.path.insert(0, "C:\\Users\\gps_0\\Downloads\\mrgteaching\\mrgpy")
sys.path.insert(0, "C:\\Users\\gps_0\\Downloads\\mrgteaching")
from pytransform import pyarmor_runtime
pyarmor_runtime()
""" ******************************************************************************** """


# MRG packages
from femcalc import meshgrid
import femcalc.meshgrid.graphics
from femcalc import fem
import pde
import mrgpy
import alip


PACKAGE_DIR = os.path.dirname(os.path.abspath(mrgpy.__file__))
    # DEMO_DIR = os.path.join(PACKAGE_DIR, 'examples')
INPUT_DATA_DIR = os.path.join(PACKAGE_DIR, '..', 'data', 'input')
OUTPUT_DATA_DIR = os.path.join(PACKAGE_DIR, '..', 'data', 'output')


class ProcessComputeElementaryMatrix(object):
    def __init__(self, what, source_term=None):
        self.source_term = source_term
        self.what = what

    def __call__(self, mesh, elem, edofs, dtype=numpy.float64):
        if mesh.elem_geotype[elem] == meshgrid.ElemType.LINE:
            dim = 3  # put 2 for 2D -> 2D
            nodes = mesh.node_coord[edofs, 0:dim]
            if self.what == 'K':
                D = numpy.identity(dim, dtype=dtype)
                K = fem.linefem.calcEllipticTerm(nodes, dtype=dtype)
                return [K]
            elif self.what == 'M':
                M = fem.linefem.calcMassTerm(nodes, dtype=dtype)
                return [M]
            elif self.what == 'F':
                f = self.source_term[edofs]
                F = fem.linefem.calcSourceTerm(nodes, f=f, dtype=dtype)
                return [F]
        elif mesh.elem_geotype[elem] == meshgrid.ElemType.QUAD:
            dim = 3  # put 2 for 2D -> 2D
            nodes = mesh.node_coord[edofs, 0:dim]
            if self.what == 'K':
                D = numpy.identity(dim, dtype=dtype)
                K = fem.quadfem.calcEllipticTerm(nodes, ngpt=2, constitutive_mat=D, dtype=dtype)
                return [K]
            elif self.what == 'M':
                M = fem.quadfem.calcMassTerm(nodes, ngpt=2, dtype=dtype)
                return [M]
            elif self.what == 'F':
                f = self.source_term[edofs]
                F = fem.quadfem.calcSourceTerm(nodes, ngpt=2, f=f, dtype=dtype)
                return [F]
        elif mesh.elem_geotype[elem] == meshgrid.ElemType.HEXAHEDRON:
            dim = 3  # put 2 for 2D -> 2D
            nodes = mesh.node_coord[edofs, 0:dim]
            if self.what == 'K':
                D = numpy.identity(dim, dtype=dtype)
                K = fem.hexfem.calcEllipticTerm(nodes, ngpt=2, constitutive_mat=D, dtype=dtype)
                return [K]
            elif self.what == 'M':
                M = fem.hexfem.calcMassTerm(nodes, ngpt=2, dtype=dtype)
                return [M]
            elif self.what == 'F':
                f = self.source_term[edofs]
                F = fem.hexfem.calcSourceTerm(nodes, ngpt=2, f=f, dtype=dtype)
                return [F]
        else:
            return None


def femReadInputParameters(filename):
    """Read input parameters from file in json format.

    References: link to control the correctness of a json file format
    https://jsonformatter.curiousconcept.com/
    """
    selfsig = lambda x: x.__module__ + '.' + x.__name__
    mygpse = selfsig(femReadInputParameters) + '()'
    mygps = '#  '
    print(f'Traceback {mygpse}: Reading file (json)')

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Error: Could not find {filename}")

    import json
    with open(filename) as json_file:
        params_json = json.load(json_file)

    print(params_json)

    from munch import munchify
    params_munch = munchify(params_json)

    import collections
    stg = 'filebasename' + ' ' \
    'boundaries' + ' ' \
    'wavenumber' + ' ' \
    'is_analytic_solution' + ' ' \
    'is_finite_element_analysis' + ' ' \
    'is_modal_analysis' + ' ' \
    'storage' + ' ' \
    'solver_type' + ' ' \
    'solver_name' + ' ' \
    'maximum_iteration_number' + ' ' \
    'residual_threshold' + ' ' \
    'is_matrix_analysis' + ' ' \
    'is_plot'
    Case = collections.namedtuple('Case', stg)
    params_namedtuples = [Case(a, b, c, d, e, f, g, h, i, j, k, l, m) \
                          for a in params_munch.mesh_data.filebasename \
                          for b in params_munch.mesh_data.boundaries \
                          for c in params_munch.eqn_data.wavenumber \
                          for d in params_munch.eqn_data.is_analytic_solution \
                          for e in params_munch.eqn_data.is_finite_element_analysis \
                          for f in params_munch.eqn_data.is_modal_analysis \
                          for g in params_munch.fem_data.storage \
                          for h in params_munch.solver_data.solver_type \
                          for i in params_munch.solver_data.solver_name \
                          for j in params_munch.solver_data.maximum_iteration_number \
                          for k in params_munch.solver_data.residual_threshold \
                          for l in params_munch.solver_data.is_matrix_analysis \
                          for m in params_munch.solver_data.is_plot \
                          ]
    return params_namedtuples

def complex_normalize(m):

    # m need to be a matrix
    a, b = m.shape
    for i in range(a):
        norma = 0
        d = m[i].flat
        t = list(d)
        for l in t:
            norma += (l * l.conjugate())
        norma = norma ** 0.5
        m[i] /= norma

    return m

def femFiniteElementMethod():
    # ***************************************************
    # WARNING: the code does not work for alpha \neq 1
    # since in this case we do not have second order partial
    # differential equations!

    selfsig = lambda x: x.__module__ + '.' + x.__name__
    mygpse = selfsig(femFiniteElementMethod) + '()'
    mygps = '#  '

    input_path = OUTPUT_DATA_DIR
    output_path = OUTPUT_DATA_DIR

    # ----------------------------------------------------------------------
    # -- read input parameters
    # ----------------------------------------------------------------------
    print(f'Traceback {mygpse}: Reading input parameters.')
    filename = 'zejson.txt'
    pathname = OUTPUT_DATA_DIR
    filename = os.path.join(pathname, filename)
    print(f'Logging: input file {filename}')
    params_namedtuples = femReadInputParameters(filename)

    # ======================================================================
    # == loop on input parameters
    # ======================================================================
    for params in params_namedtuples:
        print(f'{mygps}input parameter')
        for key, val in params._asdict().items():
            print(f'{mygps}{key} : {val}')

        str_wavenumber = f"{params.wavenumber}".replace('.','_')

        stg = f"{params.filebasename}-w{str_wavenumber}-{params.storage}-{params.solver_name}-{params.residual_threshold}"
        print(f'Logging: output files format {stg}xxx_yyy.zzz')

        # ----------------------------------------------------------------------
        # -- set analytic solution
        # ----------------------------------------------------------------------
        if params.is_analytic_solution is True:
            print(f'Traceback {mygpse}: Setting analytic solution.')
            theta = math.pi/2.
            phi = math.pi/4.
            wavenumber = params.wavenumber[0]
            exact_solution = pde.symfunc.FuncPlaneWave([theta, phi, wavenumber])
            problem = pde.bvp.EqnSecondOrderHyperCube(exact_solution)
        else:
            pass

        # ----------------------------------------------------------------------
        # -- read mesh and boundary meshes
        # ----------------------------------------------------------------------
        print(f'Traceback {mygpse}: Reading mesh and numbering.')
        input_filename = params.filebasename + '_ugr.vtk'
        input_filename = os.path.join(input_path, input_filename)
        domain_mesh = meshgrid.iomrg.readMrgFromVtk(input_filename)
        input_filename = params.filebasename + '_l2g.mtx'
        input_filename = os.path.join(input_path, input_filename)
        domain_l2g = sio.mmread(input_filename)
        domain_mesh.writeToStdout()

        print(f'Traceback {mygpse}: Reading boundary meshes and numbering.')
        boundary_meshes = {}
        boundary_l2gs = {}
        import itertools
        for k in itertools.count():
            try:
                input_filename = params.filebasename + "%06d" % (k,) + '_ugr.vtk'
                input_filename = os.path.join(input_path, input_filename)
                boundary_meshes[k] = meshgrid.iomrg.readMrgFromVtk(input_filename)
                input_filename = params.filebasename + "%06d" % (k,) + '_l2g'
                input_filename = os.path.join(input_path, input_filename)
                boundary_l2gs[k] = sio.mmread(input_filename)
            except:
                break

        energy = []

        # Create a dictionary with the calculated values of the alpha coefficient.
        alphas = dict()
        alphas[1] = complex(10 - 10j)
        alphas[2] = complex(11.20021977 - 11.30838268j)
        alphas[3] = complex(13.68 - 13.88j)
        alphas[4] = complex(15.76 - 16.07j)
        alphas[5] = complex(16.73 - 18.79j)
        alphas[6] = complex(19.22 - 19.76j)
        alphas[7] = complex(20.71 - 21.39j)
        alphas[8] = complex(22.20 - 22.91j)
        alphas[9] = complex(23.37 - 24.35j)
        alphas[10] = complex(24.66 - 25.89j)
        alphas[11] = complex(25.75 - 27.11j)
        alphas[12] = complex(26.76 - 28.29j)
        alphas[13] = complex(29.71 - 29.55j)
        alphas[14] = complex(28.84 - 30.79j)
        alphas[15] = complex(29.71 - 31.67j)
        alphas[16] = complex(30.64 - 33.00j)
        alphas[17] = complex(31.33 - 34.14j)
        alphas[18] = complex(31.40 - 35.18j)
        alphas[19] = complex(33.15 - 36.26j)
        alphas[20] = complex(33.97 - 36.98j)
        alphas[21] = complex(34.86956027 - 39.2482969j)
        alphas[22] = complex(35.55972381 - 38.96471719j)
        alphas[23] = complex(36.37735743 - 39.81218064j)
        alphas[24] = complex(36.9195351 - 41.19620619j)
        alphas[25] = complex(37.68843461 - 36.60165221j)
        alphas[26] = complex(37.98046679 - 41.2088303j)
        alphas[27] = complex(38.8753766 - 43.58178879j)
        alphas[28] = complex(38.99997144 - 44.1653056j)
        alphas[29] = complex(40.17592501 - 45.65052768j)
        alphas[30] = complex(40.68603689 - 45.70676429j)
        alphas[31] = complex(40.9469396 - 46.64397079j)
        alphas[32] = complex(42.08582465 - 47.93221063j)
        alphas[33] = complex(43.14617497 - 47.85799093j)
        alphas[34] = complex(42.72156373 - 48.35431646j)
        alphas[35] = complex(43.94432592 - 50.44014222j)
        alphas[36] = complex(43.96220158 - 51.79622084j)
        alphas[37] = complex(44.47626072 - 50.64170903j)
        alphas[38] = complex(46.84834443 - 54.16328675j)
        alphas[39] = complex(44.05685953 - 51.24287772j)
        alphas[40] = complex(45.05681276 - 54.54796475j)
        alphas[41] = complex(46.10785833 - 54.83860194j)
        alphas[42] = complex(39.12186258 - 57.18538701j)
        alphas[43] = complex(46.45643745 - 55.9657702j)
        alphas[44] = complex(45.71619129 - 56.34784214j)
        alphas[45] = complex(48.14744402 - 58.28488724j)
        alphas[46] = complex(48.88565931 - 59.09008603j)
        alphas[47] = complex(53.00385732 - 64.58785647j)
        alphas[48] = complex(48.77352693 - 61.95009813j)
        alphas[49] = complex(49.79718704 - 62.13608789j)
        alphas[50] = complex(54.87115253 - 65.42382656j)
        alphas[51] = complex(43.16609097 - 57.10496921j)
        alphas[52] = complex(43.2735739 - 54.64578529j)
        alphas[53] = complex(50.78186549 - 63.56826209j)
        alphas[54] = complex(51.88578646 - 65.31204285j)
        alphas[55] = complex(51.67626264 - 64.51732573j)
        alphas[56] = complex(50.9099199 - 67.72956163j)
        alphas[57] = complex(52.34028381 - 65.9545511j)
        alphas[58] = complex(52.301865 - 64.5711245j)
        alphas[59] = complex(53.59118949 - 68.25973657j)
        alphas[60] = complex(52.79285883 - 67.71639993j)

        # Pass the range and the code will iterate all the integer wavenumbers inside this range
        interval_w = [i for i in numpy.arange(params.wavenumber[0], params.wavenumber[1])]

        for k in interval_w:
            print("#################")
            print("WAVENUMBER: " + str(k))
            print("#################")
            # ----------------------------------------------------------------------
            # -- finite element method (assembly)
            # ----------------------------------------------------------------------
            print(f'Traceback {mygpse}: Finite element method (assembly).')
            # assembler in domain
            domain_assembler = fem.assembly.MrgAssembler(domain_mesh, dtype=numpy.complex128)
            # preassembly stiffness in domain
            ProcessComputeGradP1xGradP1ElementaryMatrix = ProcessComputeElementaryMatrix('K', None)
            domain_assembler.preassemble(['K'], [], ProcessComputeGradP1xGradP1ElementaryMatrix)
            # preassembly mass in domain
            ProcessComputeP1xP1ElementaryMatrix = ProcessComputeElementaryMatrix('M', None)
            domain_assembler.preassemble(['M'], [], ProcessComputeP1xP1ElementaryMatrix)
            # preassembly rhs in domain
            if params.is_analytic_solution is True:
                domain_alpha = 1.
                domain_beta = k ** 2
                domain_rhseqn = problem.evalSourceTermOn(domain_mesh.node_coord, domain_alpha, domain_beta)
            else:
                domain_alpha = 1.
                domain_beta = k ** 2
                domain_rhseqn = numpy.zeros((domain_mesh.numb_node,), dtype=numpy.complex128)
            ProcessComputeP1xFElementaryRhs = ProcessComputeElementaryMatrix('F', domain_rhseqn)
            domain_assembler.preassemble([], ['F'], ProcessComputeP1xFElementaryRhs)
            # assembly stiffness in domain
            domain_mat_k = domain_assembler.assemble('K', params.storage)
            # assembly mass in domain
            domain_mat_m = domain_assembler.assemble('M', params.storage)
            # assembly rhs in domain
            domain_rhs = domain_assembler.assemble('F', params.storage)

            id, dirichlet_id, neumann_id, robin_id = 0, 0, 0, 0
            dirichlet_dict = dict()
            neumann_alpha, neumann_beta, neumann_rhs = [], [], []
            robin_alpha, robin_beta, robin_rhs, robin_mat_m = [], [], [], []

            for key, value in params.boundaries:
                if value == 'dirichlet':
                    bnd_mesh = boundary_meshes[key]
                    bnd_l2g = boundary_l2gs[key]
                    for id_node in range(bnd_mesh.numb_node):
                        gid_node = bnd_l2g[id_node, 0]
                        x, y, z = bnd_mesh.node_coord[id_node, :]
                        if params.is_analytic_solution is True:
                            dirichlet_dict[gid_node] = exact_solution.toNumeric()(x, y, z)
                        else:
                            dirichlet_dict[gid_node] = complex(1., 0.)
                    dirichlet_id += 1
                elif value == 'neumann':
                    bnd_mesh = boundary_meshes[key]
                    bnd_l2g = boundary_l2gs[key]
                    # assembler on neumann
                    neumann_assembler = fem.assembly.MrgAssembler(bnd_mesh, dtype=numpy.complex128)
                    # preassembly rhs on neumann
                    if params.is_analytic_solution is True:
                        neumann_alpha.append(complex(1., 0.))
                        neumann_beta.append(complex(1., 0.))
                        neumann_rhseqn = problem.evalBoundaryTermOn(value, key, bnd_mesh.node_coord,
                                                                    neumann_alpha[neumann_id], neumann_beta[neumann_id])
                    else:
                        neumann_alpha.append(complex(1., 0.))
                        neumann_beta.append(complex(1., 0.))
                        neumann_rhseqn = numpy.zeros((bnd_mesh.numb_node,), dtype=numpy.complex128)
                    ProcessComputeP1xFElementaryRhs = ProcessComputeElementaryMatrix('F', neumann_rhseqn)
                    neumann_assembler.preassemble([], ['F'], ProcessComputeP1xFElementaryRhs)
                    # remap, i.e., change numbering
                    neumann_assembler.remap(bnd_l2g.reshape(-1, ), domain_assembler.total_ndofs)
                    # assembly rhs on neumann
                    neumann_rhs.append(neumann_assembler.assemble('F', params.storage))
                    neumann_id += 1
                elif value == 'robin':
                    bnd_mesh = boundary_meshes[key]
                    bnd_l2g = boundary_l2gs[key]
                    # assembler on robin
                    robin_assembler = fem.assembly.MrgAssembler(bnd_mesh, dtype=numpy.complex128)
                    # preassembly mass on robin
                    ProcessComputeP1xP1ElementaryMatrix = ProcessComputeElementaryMatrix('M', None)
                    robin_assembler.preassemble(['M'], [], ProcessComputeP1xP1ElementaryMatrix)
                    # preassembly rhs on robin
                    if params.is_analytic_solution is True:
                        robin_alpha.append(complex(1., 0.))
                        robin_beta.append(alphas[k])
                        robin_rhseqn = problem.evalBoundaryTermOn('robin', key, bnd_mesh.node_coord, robin_alpha[robin_id],
                                                                  robin_beta[robin_id])
                    else:
                        robin_alpha.append(complex(1., 0.))
                        robin_beta.append(alphas[k])
                        robin_rhseqn = numpy.zeros((bnd_mesh.numb_node,), dtype=numpy.complex128)
                    ProcessComputeP1xFElementaryRhs = ProcessComputeElementaryMatrix('F', robin_rhseqn)
                    robin_assembler.preassemble([], ['F'], ProcessComputeP1xFElementaryRhs)
                    # remap, i.e., change numbering
                    robin_assembler.remap(bnd_l2g.reshape(-1, ), domain_assembler.total_ndofs)
                    # assembly mass on robin
                    robin_mat_m.append(robin_assembler.assemble('M', params.storage))
                    # assembly rhs on robin
                    robin_rhs.append(robin_assembler.assemble('F', params.storage))
                    robin_id += 1
                id += 1

            # variational formulation
            cmplx1 = complex(1., 0.)
            lhs = (- domain_alpha * domain_mat_k + cmplx1 * domain_beta * domain_mat_m)
            for id in range(len(robin_mat_m)):
                mat = numpy.asarray(robin_mat_m[id])
                beta = robin_beta[id]
                lhs += - beta * mat

            rhs = domain_rhs
            if params.storage == 'dense':
                for mat in neumann_rhs:
                    rhs -= numpy.asarray(mat)
                for mat in robin_rhs:
                    rhs -= numpy.asarray(mat)
            else:
                a = rhs.tocoo()
                for mat in neumann_rhs:
                    b = mat.tocoo()
                    a -= b
                for mat in robin_rhs:
                    b = mat.tocoo()
                    a -= b
                rhs = a.tocsr()

            lhs, rhs = fem.assembly.applyDirichletBC(params.storage, lhs, rhs, dirichlet_dict, inplace=False)

            # ----------------------------------------------------------------------
            # -- finite element method (solve linear system)
            # ----------------------------------------------------------------------
            print(f'Traceback {mygpse}: Finite element method (solve linear system).')
            if params.solver_type == 'direct':
                if params.storage == 'dense':
                    sol = scipy.linalg.solve(lhs, rhs)
                else:
                    sol = scipy.sparse.linalg.spsolve(lhs, rhs)
            else:
                print(f'Error:')
                print(f'  In function {mygpse} functionality is not implemented.')
                exit(-1)

            # ----------------------------------------------------------------------
            # -- save matrix, right hand side, solution
            # ----------------------------------------------------------------------
            print(f'Traceback {mygpse}: Saving matrix, right hand side, solution.')
            if params.storage == 'dense':
                rhsd = rhs
                sold = sol
            else:
                rhsd = rhs.todense()
                sold = sol.reshape(sol.size, 1)

            energy.append(LA.norm(sold))

            domain_mesh.addNodeField('SOLUTION_' + str(k), sol)
            output_filename = stg + '_ugr_' + str(k) + '.vtk'
            output_filename = os.path.join(output_path, output_filename)
            meshgrid.iomrg.writeVtkFromMrg(domain_mesh, output_filename)

            output_filename = stg + '_lhs_' + str(k) + params.storage
            output_filename = os.path.join(output_path, output_filename)
            sio.mmwrite(output_filename, lhs)

            output_filename = stg + '_rhs_' + str(k) + params.storage
            output_filename = os.path.join(output_path, output_filename)
            sio.mmwrite(output_filename, rhsd)

            output_filename = stg + '_sol_' + str(k) + params.storage
            output_filename = os.path.join(output_path, output_filename)
            sio.mmwrite(output_filename, sold)

            output_filename = stg + '_solabs_' + str(k)
            output_filename = os.path.join(output_path, output_filename)
            myarray = numpy.squeeze(numpy.absolute(numpy.asarray(sold)), axis=1)
            meshgrid.graphics.plotting.contourf(domain_mesh, myarray, output_filename, '|u|', dpi=600)

            output_filename = stg + '_solre_' + str(k)
            output_filename = os.path.join(output_path, output_filename)
            myarray = numpy.squeeze(numpy.real(numpy.asarray(sold)), axis=1)
            meshgrid.graphics.plotting.contourf(domain_mesh, myarray, output_filename, 'Re(u)', dpi=600)

            output_filename = stg + '_solim_' + str(k)
            output_filename = os.path.join(output_path, output_filename)
            myarray = numpy.squeeze(numpy.imag(numpy.asarray(sold)), axis=1)
            meshgrid.graphics.plotting.contourf(domain_mesh, myarray, output_filename, 'Im(u)', dpi=600)

            # ----------------------------------------------------------------------
            # -- finite element method (solve eigenvalues problem)
            # ----------------------------------------------------------------------
            print(f'Traceback {mygpse}: Finite element method (solve eigenvalues problem).')
            if params.is_modal_analysis is True:
                # -- compute eigenvalues
                lambdas, u = numpy.linalg.eig(lhs.todense())
                #print(rhs, rhsd)
                #exit(1)
                # normalize eigenvectors with L2 norm
                u_norm = complex_normalize(u)
                #print(u, u.shape)
                #print(u_norm, u_norm.shape)
                #exit(1)
                #print(lambdas, u_norm)


                 # -- compute existence surface
                # existence_surface = []
                #for i in range(0,u_norm.shape[0]):
                #    integrator = fem.meshcalc.MeshIntegrator(domain_mesh, ngpt=9, ambient_dim=2)
                #    u = u_norm[:,i]
                #    u = numpy.squeeze(numpy.absolute(numpy.asarray(u)), axis=1)
                #    temp = fem.meshcalc.computeFieldIntegral(integrator, u, integrand_type='u**4', dtype=numpy.complex128)
                #    existence_surface.append(temp)
                #    print('existence_surface : ', existence_surface)

                # -- sort eigenvalues and eigenmodes
                u_norm.sort()
                lambdas.sort()

                # -- plot eigenvalues
                import alip
                output_filename = stg + '_eigs_' + str(k)
                output_filename = os.path.join(OUTPUT_DATA_DIR, output_filename)
                alip.matrices.scatter(lambdas, output_filename)

                # -- plot eigenvectors
                imin, imax = 0, u_norm.shape[1]
                for i in range(imin, imax, 1):
                    print("Eigenvector: " + str(i) + " of " + str(imax))
                    node_data = u_norm[:, i]
                    eigenvalue = complex(lambdas[i])
                    print("#  eigenvalue number %d" % (i,), ' : ', lambdas[i])
                    output_filename = stg + "__%06d__eigabs" % (i,) + '_' + str(k)
                    output_filename = os.path.join(output_path, output_filename)

                    print(type(node_data))
                    print(numpy.asarray(node_data).shape)

                    exit(1)
                    myarray = numpy.squeeze(numpy.absolute(numpy.asarray(node_data)), axis=1)
                    meshgrid.graphics.plotting.contourf(domain_mesh, myarray, output_filename, str(eigenvalue), dpi=600)

                    print(f'Traceback {mygpse}: Plotting eigenvalues.')
                    output_filename = stg + '_eigval_' + str(k)
                    output_filename = os.path.join(OUTPUT_DATA_DIR, output_filename)
                    alip.matrices.scatter(numpy.asarray([eigenvalue.real, eigenvalue.imag]), output_filename)

                # -- save eigenvalues
                print(f'Traceback {mygpse}: Saving eigenvalues.')
                output_filename = stg + '_eigs_ugr_' + str(k) + '.vtk'
                output_filename = os.path.join(OUTPUT_DATA_DIR, output_filename)
                meshgrid.iomrg.writeVtkFromMrg(domain_mesh, output_filename, binary=True)

        # Create 1x1 sub plots
        gs = gridspec.GridSpec(1, 1)

        pl.figure(figsize=(3,2))
        ax = pl.subplot(gs[0, 0])  # row 0, col 0
        pl.ylabel("|u|²")
        pl.xlabel("k")

        print(energy)
        print(interval_w, energy)
        pl.plot(interval_w, energy)

        plt.show()
        return
def demo_fem():
    femFiniteElementMethod()


if __name__ == '__main__':
    demo_fem()
