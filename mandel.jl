# Import necessary libraries
using Gridap                # Main Gridap package for finite element analysis
using Gridap.Arrays         # For Table datastructure
using Gridap.Geometry       # For mesh and geometry handling
using Gridap.FESpaces       # For finite element spaces
using Gridap.MultiField     # For coupled multi-physics problems
using Gridap.Io             # For input/output operations
using Gridap.Fields         # For field operations
using Gridap.TensorValues   # For tensor operations
using Gridap.ODEs           # For time-dependent problems
using Gridap.CellData       # For cell data operations and projection
using WriteVTK              # For VTK file output (visualization)
using GridapGmsh            # For Gmsh mesh integration

# ============================================================================
# PROBLEM DESCRIPTION
# ============================================================================
# This is a plane strain poroelasticity formulation
# In plane strain, we assume εzz = 0 (no strain in z-direction)
# but σzz ≠ 0 (stress in z-direction can exist)
# Appropriate for modeling soil/rock layers where z-dimension is constrained

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Material properties
E = 1.0e6         # Young's modulus (Pa)
nu = 0.2          # Poisson's ratio
B = 0.8           # Biot coefficient
M = 1.0e8         # Biot modulus (Pa)
k = 1.0e-12        # Permeability (m^2)
mu = 1.0e-3       # Fluid viscosity (Pa·s)

# Loading conditions
F = 1.0e3         # Compressive traction (Pa) at top boundary

# Time stepping parameters
T = 1.0          # Final time (s)
num_steps = 100   # Number of time steps
dt = T / num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
# Calculate Lamé parameters for plane strain
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter (Pa)
mu_shear = E / (2 * (1 + nu))                # Second Lamé parameter (Pa)
k_mu = k / mu                                # Hydraulic conductivity

# ============================================================================
# SETUP OUTPUT AND MESH
# ============================================================================
# Create output directory
output_dir = "results"
if !isdir(output_dir)
    mkdir(output_dir)
end

# Define internal grid using CartesianDiscreteModel
n = 20  # Number of cells per direction (matching Gmsh Transfinite setting)
domain = (0.0, 1.0, 0.0, 1.0)  # x-min, x-max, y-min, y-max
partition = (n, n)  # Number of cells in x and y directions
model = CartesianDiscreteModel(domain, partition)

# Assign boundary tags
labeling = get_face_labeling(model)
add_tag_from_tags!(labeling, "corner",[1])
add_tag_from_tags!(labeling, "bottom", [1,2,5])     # Bottom boundary (face with min y)
add_tag_from_tags!(labeling, "top", [3,4,6])      # Top boundary (face with max y)
add_tag_from_tags!(labeling, "left",[1,3,7])
add_tag_from_tags!(labeling, "right",[2,4,8])

# Export mesh for visualization
writevtk(model, "model")


# ============================================================================
# DOMAIN AND INTEGRATION SETUP
# ============================================================================
# Integration degrees
degree_u = 4  # For displacements
degree_s = 2  # For scalar fields

# Triangulation and measures
Ω = Triangulation(model)
dΩ_u = Measure(Ω, degree_u)
dΩ_s = Measure(Ω, degree_s)

# Top boundary for traction
Γ_top = BoundaryTriangulation(model, tags="top")
dΓ_top_u = Measure(Γ_top, degree_u)

# Right boundary for constraint 
Γ_right = BoundaryTriangulation(model, tags="right")

# ============================================================================
# FINITE ELEMENT SPACES
# ============================================================================
# Polynomial orders
order_u = 2  # P2 for displacement
order_s = 1  # P1 for scalars (LBB condition)

# Reference finite elements
reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
reffe_s = ReferenceFE(lagrangian, Float64, order_s)

# Base displacement space (no Dirichlet conditions yet)
δu = TestFESpace(model, reffe_u, conformity=:H1,
                 dirichlet_tags=["corner", "left", "bottom"],
                 dirichlet_masks=[(true, true), (true, false), (false, true)])


function get_dof(geometry_id, model, test_space, reffe_order, component=2)
    # Extract the identifier of the vertex located at the upper left corner of the domain
    labels = get_face_labeling(model)
    mesh_vertex_id = findall(x->x==geometry_id, labels.d_to_dface_to_entity[1])[1]

    # Extract the identifier of the cell that touches the upper left corner vertex
    topo = get_grid_topology(model)
    cells_around_vertices = Gridap.Geometry.get_faces(topo, 0, 2)
    cell_id = cells_around_vertices[mesh_vertex_id][1]

    # Extract the local vertex id of the upper left corner vertex in the cell
    cells_vertices = Gridap.Geometry.get_faces(topo, 2, 0)
    cell_vertices  = cells_vertices[cell_id]
    local_vertex_id = findall(x->x==mesh_vertex_id, cell_vertices)[1]

    # Extract the y-component DoF Id owned by the vertex located at the upper left corner of the domain
    cells_dof_ids = get_cell_dof_ids(test_space)
    cell_dof_ids = cells_dof_ids[cell_id]
    lag_ref_fe = Gridap.ReferenceFEs.LagrangianRefFE(VectorValue{2,Float64}, QUAD, reffe_order)
    face_own_dofs = Gridap.ReferenceFEs.get_face_own_dofs(lag_ref_fe)
    local_dof_ids_own_dofs = face_own_dofs[local_vertex_id]
    dof_ids_own_dofs = cell_dof_ids[local_dof_ids_own_dofs]
    dof_ids_own_dofs[component]
end


# Top left corner y dof 
master_dof = get_dof(3, model, δu, 2, 2) 
# Top right corner y dof 
slave_dof = get_dof(4, model, δu, 2, 2)  # u_y of other nodes

# Constraint setup
sDOF_to_dof = [slave_dof] 
sDOF_to_dofs = Table([[master_dof]])
sDOF_to_coeffs = Table([[1.0]])

# Constrained space
δu = FESpaceWithLinearConstraints(sDOF_to_dof, sDOF_to_dofs, sDOF_to_coeffs, δu)

u = TrialFESpace(δu, _ -> VectorValue(0.0, 0.0))

# Pressure space
δp = TestFESpace(model, reffe_s, conformity=:H1, dirichlet_tags=["right"])
p = TrialFESpace(δp, 0.0)

# Multi-field space
Y = MultiFieldFESpace([δu, δp])

# ============================================================================
# TRANSIENT TRIAL SPACES
# ============================================================================
u_t = TransientTrialFESpace(δu)
p_t = TransientTrialFESpace(δp)
X_t = MultiFieldFESpace([u_t, p_t])

# ============================================================================
# RESIDUAL FORMULATION
# ============================================================================
# Stress tensor (plane strain)
sigma(u) = 2 * mu_shear * symmetric_gradient(u) + lambda * tr(symmetric_gradient(u)) * one(TensorValue{2,2,Float64})

function m(t, (du, dp), (δu, δp))
    ∫( # Order 4 terms
        δp * B * divergence(du)
     ) * dΩ_u +
    ∫( # Order 2 terms
        δp * (1/M) * dp 
    ) * dΩ_s
end

function a(t, (u, p), (δu, δp))
    ∫( # Order 4 terms
        symmetric_gradient(δu) ⊙ sigma(u) -
        (B * divergence(δu) * p)
    ) * dΩ_u +
    ∫( # Order 2 terms
        ∇(δp) ⋅ (k_mu * ∇(p))
    ) * dΩ_s
end

function l(t, (δu, δp))
    ∫( δu ⋅ VectorValue(0.0, -F) ) * dΓ_top_u
end


# ============================================================================
# INITIAL CONDITIONS
# ============================================================================
u0 = VectorValue(0.0, 0.0)
p0 = 1000.0
uh0 = interpolate_everywhere([u0, p0], X_t(0.0))

# ============================================================================
# TRANSIENT PROBLEM SETUP
# ============================================================================
op = TransientLinearFEOperator((a, m), l, X_t, Y, constant_forms=(true, true))

# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================
# Set up the linear solver (for solving linear systems within Newton iterations)
ls = LUSolver()  # Direct LU decomposition solver

# Create the ODE solver with the nonlinear solver
Δt = dt  # Time step size
θ = 1.0  # Backward Euler scheme (θ=1.0 is fully implicit)
         # Note: θ=0.5 would be Crank-Nicolson, θ=0.0 would be forward Euler
ode_solver = ThetaMethod(ls, Δt, θ)


# ============================================================================
# SOLVE THE TRANSIENT PROBLEM
# ============================================================================
# Set time interval
t0 = 0.0  # Initial time
tF = T    # Final time

# Solve the time-dependent problem
# This returns a generator that yields (time, solution) pairs
sol = solve(ode_solver, op, t0, tF, uh0)

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================
# Create ParaView collection (.pvd file) for time-series visualization
createpvd(joinpath(output_dir, "results")) do pvd
    # Save initial state (t=0)
    u0_h, p0_h = uh0  # Extract displacement and pressure from initial solution
    pvd[0.0] = createvtk(Ω, joinpath(output_dir, "results_0.vtu"), 
                         cellfields=["displacement"=>u0_h, "pressure"=>p0_h])
    
    # Save solution at each time step
    for (tn, uhn) in sol
        println("Writing results for t = $tn")
        
        # Extract displacement and pressure from current solution
        un_h, pn_h = uhn
        
        # Create VTK file for this time step with displacement and pressure fields
        pvd[tn] = createvtk(Ω, joinpath(output_dir, "results_$(tn).vtu"), 
                           cellfields=["displacement"=>un_h, 
                                      "pressure"=>pn_h])
    end
end

# Print completion message
println("Plane strain simulation completed! Results saved in the '$output_dir' directory.")
