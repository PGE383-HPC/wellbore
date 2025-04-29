#boundary Import necessary libraries
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
E = 20.0e6         # Young's modulus (Pa)
nu = 0.2          # Poisson's ratio
B = 0.8           # Biot coefficient (coupling between fluid pressure and solid stress)
M = 1.0e9         # Biot modulus (Pa) - related to fluid and solid compressibility
k = 1.0e-3        # Permeability (m^2) - how easily fluid flows through the medium
mu = 1.0e-3       # Fluid viscosity (Pa·s)

# Loading conditions
Pb = 31.5e6
p0 = 20.0e6

# Time stepping parameters
T = 0.0005          # Final time (s)
num_steps = 3   # Number of time steps
dt = T / num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
# Calculate Lamé parameters for plane strain formulation
# For plane strain, we use the same Lamé parameters as in 3D
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter (Pa)
mu = E / (2 * (1 + nu))                      # Second Lamé parameter (shear modulus) (Pa)
k_mu = k / mu                                # Hydraulic conductivity (permeability/viscosity)

dirichlet_tags = ["top_bottom", "wellbore"]
#####################################
######## ADD YOUR CODE HERE ##########
#####################################

# -- 1. Read in the Gmsh geometry and build the Gridap model
model = GmshDiscreteModel("wellbore.geo")
# extract the volume and boundary triangulations
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
Γ = BoundaryTriangulation(model)
dΓ = Measure(Γ, 1)

# -- 2. Define the finite element spaces
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)

# displacement space: H1-vector, u = 0 on top_bottom
V = FESpace(
  model, reffe;
  conformity = :H1,
  dirichlet_tags = ["top_bottom"],
  dirichlet_data = x -> VectorValue(0.0, 0.0)
)

# pressure space: H1-scalar, p = Pb on wellbore
Q = FESpace(
  model, reffe;
  conformity = :H1,
  dirichlet_tags = ["wellbore"],
  dirichlet_data = x -> Pb
)

# coupled multi‐field space
Y = MultiFieldFESpace([V, Q])

# helper to extract trial/test functions
(u,  p) = get_trial_fields(Y)
(v,  q) = get_test_fields(Y)

# -- 3. Initialize previous‐step solution (u=0, p=p0 everywhere)
uh_old = interpolate_everywhere(x -> VectorValue(0.0,0.0), V)
ph_old = interpolate_everywhere(x -> p0,           Q)

# -- 4. Weak forms
# symmetric gradient
ε(u) = sym(∇(u))

for step in 1:num_steps
  t = step*dt

  # Bilinear form
  a((u,p),(v,q)) = ∫( 
      # mechanics: linear elasticity minus Biot coupling
      inner(2*mu*ε(u) + lambda*div(u)*one(SymmetricTensorValue{2,Float64}) - B*p*one(SymmetricTensorValue{2,Float64}), ε(v))
      # fluid: compressibility + flow
      + (B*div(u) + (1/M)*p)*q
      + dt*k_mu*(∇(p)⋅∇(q))
    )*dΩ

  # Linear form (from previous time step)
  l((v,q)) = ∫(
      (B*div(uh_old) + (1/M)*ph_old)*q
    )*dΩ

  # assemble and solve
  op = AffineFEOperator(a, l, Y, Y)
  sol = solve(op)
  uh, ph = sol[1], sol[2]

  # update history
  uh_old .= uh
  ph_old .= ph

  # write results for ParaView
  writevtk(
    Ω, "results/step$(step)";
    cellfields = ["displacement" => uh, "pressure" => ph]
  )
end
