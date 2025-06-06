# Import necessary libraries
using Gridap                # Main Gridap package for finite element analysis
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
# This is appropriate for modeling soil/rock layers where the z-dimension is 
# constrained but stresses can develop in that direction

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Material properties
E = 1.0e6         # Young's modulus (Pa)
nu = 0.2          # Poisson's ratio
B = 0.8           # Biot coefficient (coupling between fluid pressure and solid stress)
M = 1.0e9         # Biot modulus (Pa) - related to fluid and solid compressibility
k = 1.0e-3        # Permeability (m^2) - how easily fluid flows through the medium
mu = 1.0e-3       # Fluid viscosity (Pa·s)

# Loading conditions
F = 1.0e3         # Compressive traction (Pa) applied at the top boundary

# Time stepping parameters
T = 10.0          # Final time (s)
num_steps = 100   # Number of time steps
dt = T / num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
# Calculate Lamé parameters for plane strain formulation
# For plane strain, we use the same Lamé parameters as in 3D
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter (Pa)
mu = E / (2 * (1 + nu))                      # Second Lamé parameter (shear modulus) (Pa)
k_mu = k / mu                                # Hydraulic conductivity (permeability/viscosity)

# Note: The plane strain modulus (E') can be calculated as:
# E_plane_strain = E / (1 - nu^2)
# But we'll use lambda and mu directly in our formulation as they appear
# directly in the stress-strain relationship

# ============================================================================
# SETUP OUTPUT AND MESH
# ============================================================================
# Create output directory if it doesn't exist
output_dir = "results"
if !isdir(output_dir)
    mkdir(output_dir)
end

# Load the Gmsh mesh from file
# The mesh should be a square domain with properly tagged boundaries
model = GmshDiscreteModel("square.msh")

# Define boundary tags for applying boundary conditions
# These tags should match the physical groups defined in the Gmsh file
dirichlet_tags = ["top", "bottom", "left", "right"]

# Export the mesh for visualization
writevtk(model, "model")  # Save model for visualization in ParaView

# Print information about boundary entities for debugging
# This helps verify that boundary conditions will be applied to the correct entities
labels = get_face_labeling(model)
for tag in dirichlet_tags
    println("Entities tagged as $tag: ", findall(labels.tag_to_name .== tag))
end

# ============================================================================
# DOMAIN AND INTEGRATION SETUP
# ============================================================================
# Set up integration degree for numerical quadrature
degree_u = 4  # Quadrature order for displacements
degree_s = 2  # Quadrature order for scalar fields

# Create triangulation and integration measures
Ω = Triangulation(model)           # Domain triangulation
dΩ_u = Measure(Ω, degree_u)        # Volume integration measure (for displacements)
dΩ_s = Measure(Ω, degree_s)        # Volume integration measure (for all other scalar fields)

# Create special measure for the top boundary where traction is applied
Γ_top = BoundaryTriangulation(model, tags="top")  # Extract top boundary
dΓ_top_u = Measure(Γ_top, degree_u) # Integration measure for top boundary (for displacements)
dΓ_top_s = Measure(Γ_top, degree_s) # Integration measure for top boundary (for scalars)

# ============================================================================
# FINITE ELEMENT SPACES
# ============================================================================
# Define polynomial orders for the mixed formulation
order_u = 2  # P2 (quadratic) elements for displacement
order_s = 1  # P1 (linear) elements for scalars - satisfies LBB condition

# Create reference finite elements
reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)  # Vector-valued for displacement
reffe_s = ReferenceFE(lagrangian, Float64, order_s)                 # Scalar-valued for scalars 

# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================
# Displacement space with Dirichlet BC on bottom (fixed in y-direction)
δu = TestFESpace(model, reffe_u, conformity=:H1, 
                 dirichlet_tags=["bottom"])
u = TrialFESpace(δu, x -> VectorValue(0.0, 0.0))  # Zero displacement at bottom boundary

# Pressure space with Dirichlet BC on left and right sides (drained boundaries)
δp = TestFESpace(model, reffe_s, conformity=:H1, dirichlet_tags=["left", "right"])
p = TrialFESpace(δp, 0.0)  # Zero pressure at left and right boundaries

# Lagrange multiplier field
δλ = TestFESpace(Γ_top, reffe_s, conformity=:L2, dirichlet_tags=["left", "right"])
λ = TrialFESpace(δλ, 0.0)  # Zero pressure at left and right boundaries

# Create multi-field space for the coupled problem
Y = MultiFieldFESpace([δu, δp])  # Combined test space for displacement and pressure



# ============================================================================
# CONSTITUTIVE EQUATIONS
# ============================================================================
# Function to calculate the in-plane stress tensor (σxx, σxy, σyx, σyy)
function sigma(u)
    # In plane strain, εzz = 0 but σzz ≠ 0
    # Calculate the strain tensor from displacement gradient
    ε = symmetric_gradient(u)  # ε = (∇u + (∇u)ᵀ)/2
    
    # Identity tensor in 2D
    I = TensorValue(1.0, 0.0, 0.0, 1.0)
    
    # Plane strain stress-strain relationship
    # For plane strain, the constitutive equation is the same as 3D
    # but with the constraint that εzz = 0
    # The 2D stress tensor is: σ = λ tr(ε) I + 2μ ε
    # where tr(ε) = εxx + εyy (since εzz = 0)
    # This returns the 2x2 stress tensor containing σxx, σxy, σyx, σyy
    return lambda * tr(ε) * I + 2 * mu * ε
end

# Function to calculate out-of-plane stress (σzz) for visualization
# This is needed because in plane strain, σzz is non-zero even though εzz = 0
function sigma_zz(u)
    # Calculate the strain tensor
    ε = symmetric_gradient(u)
    
    # In plane strain: σzz = λ(εxx + εyy) = λ·tr(ε)
    # This is derived from the 3D constitutive equation with εzz = 0
    return lambda * tr(ε)
end

# ============================================================================
# INITIAL CONDITIONS
# ============================================================================
# Define initial conditions for the problem
u0 = VectorValue(0.0, 0.0)  # Zero initial displacement
p0 = 0.0                    # Zero initial pressure

# ============================================================================
# TRANSIENT TRIAL SPACES
# ============================================================================
# Create transient trial spaces for time-dependent problem
# For this problem, our boundary conditions don't change with time,
# but we still need to use TransientTrialFESpace for the time integration
u_t = TransientTrialFESpace(δu)  # Transient displacement space
p_t = TransientTrialFESpace(δp)  # Transient pressure space

# Combine the transient spaces into a multi-field space
X_t = MultiFieldFESpace([u_t, p_t])

function res_Ω(t, (u, p), (δu, δp)) 
  ∫(
    # Volume terms (poroelasticity equations)
    symmetric_gradient(δu) ⊙ sigma(u) -
    (B * divergence(δu) * p) +
    δp * (1/M) * ∂t(p) +
    ∇(δp) ⋅ (k_mu * ∇(p)) +
    δp * B * divergence(∂t(u))
   ) * dΩ 
end

function res_Γ(t, (u, λ), (δu, δλ)) 
  ∫(
    # Lagrange multiplier terms for u_y = constant on top boundary
    λ * ∇(δu ⋅ VectorValue(0.0, 1.0)) ⋅ VectorValue(1.0, 0.0) +
    δλ * ∇(u ⋅ VectorValue(0.0, 1.0)) ⋅ VectorValue(1.0, 0.0)
   ) * dΓ_top -
  ∫(
    δu ⋅ VectorValue(0.0, -F)  # Compressive traction
   ) * dΓ_top
end

# ============================================================================
# WEAK FORM
# ============================================================================
# Bilinear form a(u,p,δu,δp) - represents the weak form of the PDE system
a(t, (u,p), (δu,δp)) = ∫( 
    # Solid mechanics term: stress-strain relationship
    symmetric_gradient(δu) ⊙ sigma(u) - 
    
    # Coupling term 1: effect of fluid pressure on solid (Biot coupling)
    (B * divergence(δu) * p) +
    
    # Fluid storage term: time derivative of pressure
    δp * (1/M) * ∂t(p) + 
    
    # Fluid diffusion term: Darcy's law
    ∇(δp) ⋅ (k_mu * ∇(p)) +
    
    # Coupling term 2: effect of solid deformation on fluid (Biot coupling)
    δp * B * divergence(∂t(u)) 
) * dΩ

# Linear form l(δu,δp) - represents external forces/sources
l(t, (δu,δp)) = ∫( 
    # Traction force applied at the top boundary
    δu ⋅ VectorValue(0.0, -F)  # Negative F for compression in y-direction
) * dΓ_top 

# Residual form for the nonlinear solver
res(t, (u, p, λ), (δu, δp, δλ)) = res_Ω(t, (u, p), (δu, δp)) + res_Γ(t, (u, λ), (δu, δλ))

# ============================================================================
# TRANSIENT PROBLEM SETUP
# ============================================================================
# Create the transient finite element operator from the residual
op = TransientFEOperator(res, X_t, Y)

# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================
# Set up the linear solver (for solving linear systems within Newton iterations)
ls = LUSolver()  # Direct LU decomposition solver

# Set up the nonlinear solver (Newton's method)
nls = NLSolver(ls, method=:newton, iterations=10, show_trace=false)

# Create the ODE solver with the nonlinear solver
Δt = dt  # Time step size
θ = 1.0  # Backward Euler scheme (θ=1.0 is fully implicit)
         # Note: θ=0.5 would be Crank-Nicolson, θ=0.0 would be forward Euler
ode_solver = ThetaMethod(nls, Δt, θ)

# ============================================================================
# INITIAL SOLUTION
# ============================================================================
# Interpolate the initial conditions onto the FE spaces
uh0 = interpolate_everywhere([u0, p0], X_t(0.0))

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
