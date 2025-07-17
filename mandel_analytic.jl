using Roots  # For finding roots of the transcendental equation
using Plots  # For visualization

# Material and problem parameters
function init_params()
  Dict(
    "a"   => 1.0,   # Half-width of the slab (m)
    "b"   => 1.0,   # Half-height of the slab (m)
    "F"   => 2, # Applied load (Pa)
    "E"   => 10,  # Young's modulus (Pa)
    "ν"   => 0.2,   # Poisson's ratio
    "κ"   => 1.0, # Permeability (m²)
    "μ"   => 1.0,  # Fluid viscosity (Pa·s)
    "ϕ"   => 0.2,   # Porosity
    "K_f" => 4.54, # Fluid bulk modulus
    "Kₛ" => 51.85, # Fluid bulk modulus
  )
end

function check_poisson_ratio(ν::Real)
  if ν >= 0.5 || ν < -1
    error("Poisson’s ratio must be in the range (-1, 0.5) for physical materials")
  end
  nothing
end

function compute_shear_modulus(params::Dict)
  E = params["E"]
  ν = params["ν"]
  check_poisson_ratio(ν)
  params["G"] = E / (2 * (1 + ν))  # Shear modulus
  nothing
end

function compute_lame_parameter(params::Dict)
  E = params["E"]
  ν = params["ν"]
  check_poisson_ratio(ν)
  params["λ"] = E * ν / ((1 + ν) * (1 - 2 * ν)) #Lame parameter
  nothing
end

function compute_consolidation_coefficient(params::Dict)
  if !haskey(params, "G")
    compute_shear_modulus(params)
  end
  if !haskey(params, "K")
    compute_bulk_modulus(params)
  end
  if !haskey(params, "α")
    compute_biot_coefficient(params)
  end
  if !haskey(params, "M")
    compute_biot_modulus(params)
  end
  κ  = params["κ"]
  G  = params["G"]
  μ  = params["μ"]
  K  = params["K"]
  M  = params["M"]
  α  = params["α"]
  params["c"] = κ / μ * (K + 4 / 3 * G) * M / (K + 4 / 3 * G + α^2 * M) # Consolidation coefficient
  nothing
end


function compute_solid_bulk_modulus(params::Dict)

  if haskey(params, "Kₛ")
    return nothing 
  end

  if !haskey(params, "K")
    compute_bulk_modulus(params)
  end

  if !haskey(params, "α")
    error("α must be provided to compute Kₛ")
  end

  α = params["α"]

  if α >= 1
      error("α = 1 implies infinite Kₛ; handle this case separately or ensure α < 1")
  end

  K = params["K"]
  params["Kₛ"] = K / (1 - α)
  nothing
end

function compute_bulk_modulus(params::Dict)
  ν = params["ν"]
  check_poisson_ratio(ν)
  E = params["E"]

  params["K"] = E / (3 * (1 - 2 * ν))
  nothing
end

function compute_biot_coefficient(params::Dict)
  if haskey(params, "α")
    return nothing 
  end
  if !haskey(params, "K")
    compute_bulk_modulus(params)
  end
  if !haskey(params, "Kₛ")
    error("Kₛ must be provided to compute Biot's coefficient.") 
  end
  K = params["K"]
  Kₛ = params["Kₛ"]
  params["α"] = 1 - K / Kₛ
  nothing
end

function compute_skemptons_coefficient(params::Dict)

  if !haskey(params, "K")
    compute_bulk_modulus(params)
  end

  if !haskey(params, "α")
    compute_biot_coefficient(params)
  end

  if !haskey(params, "M")
    compute_biot_modulus(params)
  end

  ϕ = params["ϕ"]
  K = params["K"]
  α = params["α"]
  M = params["M"]
  params["B"] = α * M / (K + α^2 * M)
  nothing
end

function compute_biot_modulus(params::Dict)
  if haskey(params, "M")
    return nothing 
  end
  if !haskey(params, "α")
    compute_biot_coefficient(params)
  end
  if !haskey(params, "Kₛ")
    compute_solid_bulk_modulus(params)
  end

  Kₛ = params["Kₛ"]
  K_f = params["K_f"]
  α = params["α"]
  ϕ = params["ϕ"]
  params["M"] = 1 / ((α - ϕ) / Kₛ + ϕ / K_f) 
  nothing
end

function compute_undrained_params(params::Dict)
  if !haskey(params, "α")
    compute_biot_coefficient(params)
  end
  if !haskey(params, "M")
    compute_biot_modulus(params)
  end
  if !haskey(params, "B")
    compute_skemptons_coefficient(params)
  end
  K = params["K"]
  α = params["α"]
  M = params["M"]
  B = params["B"]
  ν = params["ν"]
  params["Kᵤ"] = K + α^2 * M
  νᵤ = (3*ν + B * α * (1 - 2 * ν)) / (3 - B * α * (1 - 2 * ν))
  check_poisson_ratio(νᵤ)
  params["νᵤ"] = νᵤ
  nothing
end

function get_params()
  params = init_params()
  compute_lame_parameter(params)

  if haskey(params, "Kₛ")
    compute_biot_coefficient(params)
  elseif haskey(params, "α")
    compute_solid_bulk_modulus(params)
  else
    error("One of α or Kₛ must be provided.")
  end

  compute_undrained_params(params)
  compute_consolidation_coefficient(params)
  params
end

# Transcendental equation for roots α_n
function transcendental_eq(α, params)
    ν = params["ν"]
    νᵤ = params["νᵤ"]
    return tan(α) - ((1 - ν) / (νᵤ - ν)) * α
end

# Find roots α_n using Roots.jl
function find_roots(::Type{T}, n_roots, params) where T
    α = T[]
    for n = 1:n_roots
        guess = -π/2 + n*π - 0.001 
        root = find_zero(α -> transcendental_eq(α, params), guess)
        push!(α, root)
    end
    return α
end

find_roots(n_roots, params) = find_roots(Float64, n_roots, params)

function pore_pressure(x, t, α, params)
    a = params["a"]
    F = params["F"]
    B = params["B"]
    νᵤ = params["νᵤ"]
    c = params["c"]

    sum = 0.0
    for α_n in α
        term = sin(α_n) / (α_n - sin(α_n) * cos(α_n)) * (cos(α_n * x / a) - cos(α_n))
        sum += term * exp(-α_n^2 * c * t / a^2)
    end
    return (2 * F * B * (1 + νᵤ) / (3 * a)) * sum
end


function displacement_ux(x, t, α, params)
    a = params["a"]
    F = params["F"]
    G = params["G"]
    ν = params["ν"]
    νᵤ = params["νᵤ"]
    c = params["c"]
    coef1 = (F * ν) / (2.0 * G * a)
    coef2 = -(F * νᵤ) / (G * a)
    sum1 = 0.0
    sum2 = 0.0
    for α_n in α
        exp_t = exp(-α_n^2 * c * t / a^2)
        sum1 += sin(α_n) * cos(α_n) / (α_n - sin(α_n) * cos(α_n)) * exp_t
        sum2 += cos(α_n) / (α_n - sin(α_n) * cos(α_n)) * sin(α_n * x / a) * exp_t
    end
  return (coef1 + coef2 * sum1) * x + F / G * sum2
end

# Displacement u_y(x, y, t)
function displacement_uy(y, t, α, params)
    a = params["a"]
    F = params["F"]
    ν = params["ν"]
    νᵤ = params["νᵤ"]
    G = params["G"]
    c = params["c"]
    coef1 = -F * (1 - ν) / (2 * G * a)
    coef2 = F * (1 - νᵤ) / (G * a)
    sum = 0.0
    for α_n in α
        exp_t = exp(-α_n^2 * c * t / a^2)
        sum += sin(α_n)  * cos(α_n) / (α_n - sin(α_n) * cos(α_n)) * exp_t 
    end
    return (coef1 + coef2 * sum) * y
end

function main()
  # Compute solutions
  n_roots = 10  # Number of roots for series convergence
  params = get_params()

  roots = find_roots(n_roots, params)

  # Spatial and temporal points
  x_vals = range(0, params["a"], length=50)
  y_vals = range(0, params["b"], length=50)
  t_vals = [1e-5, 1e-4, 1e-3]  # Times to evaluate (s)

  # Optional: Plot along x at y = b for comparison
  p_vals = [pore_pressure.(x_vals, t, Ref(roots), Ref(params)) for t in t_vals]
  ux_vals = [displacement_ux.(x_vals, params["b"], t, Ref(roots), Ref(params)) for t in t_vals]
  uy_vals = [displacement_uy.(y_vals, params["b"], t, Ref(roots), Ref(params)) for t in t_vals]

  p1 = plot(xlabel="x (m)", title="Fields at y = b")
  for (i, t) in enumerate(t_vals)
    p1 = plot!(x_vals, p_vals[i], label="p, t = $t s")
  end

  p2 = plot(xlabel="x (m)", title="u_x at y = b")
  for (i, t) in enumerate(t_vals)
      p2 = plot!(x_vals, ux_vals[i], label="t = $t s")
  end

  p3 = plot(xlabel="x (m)", title="u_y at y = b")
  for (i, t) in enumerate(t_vals)
      p3 = plot!(x_vals, uy_vals[i], label="t = $t s")
  end
  plot(p1, p2, p3, layout=(3,1))
end
