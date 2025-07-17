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
using Plots
using LaTeXStrings
include("mandel_analytic.jl")

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
params = get_params()

# E = 1.0e6         # Young's modulus (Pa)
# nu = 0.2          # Poisson's ratio
# B = 0.8           # Biot coefficient
# M = 1.0e8         # Biot modulus (Pa)
# k = 1.0e-12        # Permeability (m^2)
# mu = 1.0e-3       # Fluid viscosity (Pa·s)
mu = params["μ"]
B = params["α"]
k = params["κ"]
M = params["M"]
E = params["E"]
nu = params["ν"]
nu_u = params["νᵤ"]

# Loading conditions
F = params["F"]         # Compressive traction (Pa) at top boundary

# Penalty parameter
α = 1.0e9

# Time stepping parameters
T = 0.01          # Final time (s)
num_steps = 100   # Number of time steps
dt = T / num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
# Calculate Lamé parameters for plane strain
lambda = params["λ"]  # First Lamé parameter (Pa)
mu_shear = params["G"] 
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
domain = (0.0, params["a"], 0.0, params["b"])  # x-min, x-max, y-min, y-max
partition = (50, 30)  # Number of cells in x and y directions
model = CartesianDiscreteModel(domain, partition)

# Assign boundary tags
labeling = get_face_labeling(model)
add_tag_from_tags!(labeling, "corner",[1])
add_tag_from_tags!(labeling, "bottom", [1,2,5])     # Bottom boundary (face with min y)
add_tag_from_tags!(labeling, "left",[1,3,7])
add_tag_from_tags!(labeling, "top", [3,4,6])      # Top boundary (face with max y)
add_tag_from_tags!(labeling, "right",[2,4,8])

# Export mesh for visualization
writevtk(model, "model")


# ============================================================================
# DOMAIN AND INTEGRATION SETUP
# ============================================================================
# Quadrature degree
degree = 4

# Triangulation and measures
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# Top boundary for traction
Γ_top = BoundaryTriangulation(model, tags="top")
dΓ_top = Measure(Γ_top, degree)

# Right boundary for constant pressure
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

function res_Ω(t, (u, p), (δu, δp))
    ∫( 
        symmetric_gradient(δu) ⊙ sigma(u) -
        (B * divergence(δu) * p) +
        δp * B * divergence(∂t(u)) +
        δp * (1/M) * ∂t(p) +
        ∇(δp) ⋅ (k_mu * ∇(p))
    ) * dΩ
end

function res_Γ(t, u, δu)
    ∫( δu ⋅ VectorValue(0.0, -F) ) * dΓ_top  # Traction
end

function penalty(t, u, δu)
    # Compute full gradients
    grad_u = ∇(u)    # TensorValue{2,2} for u
    grad_δu = ∇(δu)  # TensorValue{2,2} for δu

    # Basis vector for x-direction
    e_x = VectorValue(1.0, 0.0)
    #
    # Extract ∂u_y/∂x and ∂δu_y/∂x using dot products
    # Gradient of u_y is the second row of ∇u, projected onto e_x
    ∂u_y_∂x = (grad_u ⋅ VectorValue(0.0, 1.0)) ⋅ e_x  # ∂u_y/∂x
    ∂δu_y_∂x = (grad_δu ⋅ VectorValue(0.0, 1.0)) ⋅ e_x  # ∂δu_y/∂x

    ∫( 
        α * (∂u_y_∂x * ∂δu_y_∂x)
    ) * dΓ_top
end

res(t, (u, p), (δu, δp)) = res_Ω(t, (u, p), (δu, δp)) - res_Γ(t, u, δu) + penalty(t, u, δu)

# ============================================================================
# INITIAL CONDITIONS
# ============================================================================
n_roots = 20
roots = find_roots(n_roots, params)

u0(x) = VectorValue(F * nu_u * x[1] / (2 * mu_shear * params["a"]), 
                    -F * (1 - nu_u) * x[2] / (2 * mu_shear * params["a"]))
p0(x) = F * params["B"] * (1 + nu_u) / (3.0 * params["a"])
uh0 = interpolate_everywhere([u0, p0], X_t(0.0))

# ============================================================================
# TRANSIENT PROBLEM SETUP
# ============================================================================
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
# SOLVE THE TRANSIENT PROBLEM
# ============================================================================
# Set time interval
t0 = 0.0  # Initial time
tF = T    # Final time

# Solve the time-dependent problem
# This returns a generator that yields (time, solution) pairs
sol = Gridap.solve(ode_solver, op, t0, tF, uh0)

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================
# Create ParaView collection (.pvd file) for time-series visualization
# createpvd(joinpath(output_dir, "results")) do pvd
#     # Save initial state (t=0)
#     u0_h, p0_h = uh0  # Extract displacement and pressure from initial solution
#     pvd[0.0] = createvtk(Ω, joinpath(output_dir, "results_0.vtu"), 
#                          cellfields=["displacement"=>u0_h, "pressure"=>p0_h])
#     
#     # Save solution at each time step
#     for (i, (tn, uhn)) in enumerate(sol)
#         if i % 10 == 0 
#           println("Writing results for t = $tn")
#           
#           # Extract displacement and pressure from current solution
#           un_h, pn_h = uhn
#           
#           # Create VTK file for this time step with displacement and pressure fields
#           pvd[tn] = createvtk(Ω, joinpath(output_dir, "results_$(tn).vtu"), 
#                              cellfields=["displacement"=>un_h, "pressure"=>pn_h])
#         end
#     end
# end


line_points_x = [Point(x, params["b"]) for x in range(0.0, params["a"], length=100)]
line_points_y = [Point(0.0, y) for y in range(0.0, params["b"], length=100)]
x_coords = [p[1] for p in line_points_x]
y_coords = [p[2] for p in line_points_y]

# Create an empty plot
p1 = plot(xlabel=L"$x$", ylabel=L"$p$", title=L"Pressure at $y=b$")
p2 = plot(xlabel=L"$x$", ylabel=L"u_x", title=L"$x$ displacement at $y=b$")
p3 = plot(xlabel=L"$y$", ylabel=L"u_y", title=L"$y$ displacement at $x=0$")

for (tn, uhn) in sol
    # Only plot a subset of time steps to avoid overcrowding
  if any(isapprox.(tn, [1e-4, 1e-3, 1e-2]))  # Plot these time steps
        # Extract fields
        un_h, pn_h = uhn  # Adjust based on your field structure
        
        # Evaluate pressures and displacements along the lines
        pressure_values = [pn_h(point) for point in line_points_x]
        displacement_values_x = [un_h(point) for point in line_points_x]
        displacement_values_y = [un_h(point) for point in line_points_y]
        ux_values = getindex.(displacement_values_x, 1)
        uy_values = getindex.(displacement_values_y, 2)

        pressure_values_exact = pore_pressure.(x_coords, tn, Ref(roots), Ref(params))
        ux_values_exact = displacement_ux.(x_coords, tn, Ref(roots), Ref(params))
        uy_values_exact = displacement_uy.(y_coords, tn, Ref(roots), Ref(params))
        
        # Add to plot
        time = round(tn, digits=5)
        gridap_label = L"t = %$time" * ", Gridap"
        exact_label = L"t = %$time" * ", exact"

        scatter!(p1, x_coords, pressure_values, label=gridap_label, markershape=:circle, markersize=1)
        plot!(p1, x_coords, pressure_values_exact, label=exact_label)

        scatter!(p2, x_coords, ux_values, label=gridap_label, markershape=:circle, markersize=1)
        plot!(p2, x_coords, ux_values_exact, label=exact_label)

        scatter!(p3, y_coords, uy_values, label=gridap_label, markershape=:circle, markersize=1)
        plot!(p3, y_coords, uy_values_exact, label=exact_label)
    end
end

# Display the plot
p = plot(p1, p2, p3, layout=(3, 1))
display(p)
