using JuMP, GLPK
using LinearAlgebra

##
## structures
##

struct Game
  A::Matrix
  B::Matrix
  adv::Bool
end

struct CorrelatedPolicy
  a::CartesianIndex
  aprime::CartesianIndex
  w::Real
end

##
## minmax values
##
# https://theory.stanford.edu/~tim/w16/l/l10.pdf
# A simple but important observation is: the second player never needs to randomize. For
# example, suppose the row player goes first and chooses a distribution x. The column player
# can then simply compute the expected payoff of each column (the expectation with respect
# to x) and choose the best column (deterministically). If multiple columns are tied for the
# best, the it is also optimal to randomized arbitrarily among these; but there is no need for
# the player moving second to do so.
# 
# Note: maxmin value = minmax value in two player games since it can be seen as a zero sum game
# (1) https://math.stackexchange.com/q/3781785
# (2) p.20 https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-254-game-theory-with-engineering-applications-spring-2010/lecture-notes/MIT6_254S10_lec15.pdf

function minmax_value_A(G::Game)
  A = G.A
  m, n = size(A)
  model = Model(GLPK.Optimizer)
  @variable(model, y[1:n] >= 0)
  @variable(model, wA)
  @objective(model, Min, wA)
  @constraint(model, obj[i = 1:m], wA >= sum(A[i, j] * y[j] for j = 1:n))
  @constraint(model, simplex, sum(y) == 1)
  optimize!(model)
  solution_summary(model)
  return objective_value(model), value.(y)
end

function minmax_value_B(G::Game)
  B = G.B
  m, n = size(B)
  model = Model(GLPK.Optimizer)
  @variable(model, x[1:m] >= 0)
  @variable(model, wB)
  @objective(model, Min, wB)
  @constraint(model, obj[j = 1:n], wB >= sum(B[i, j] * x[i] for i = 1:m))
  @constraint(model, simplex, sum(x) == 1)
  optimize!(model)
  solution_summary(model)
  return objective_value(model), value.(x)
end

##
## maxmin values
##

function maxmin_value_A(G::Game)
  A = G.A
  m, n = size(A)
  model = Model(GLPK.Optimizer)
  @variable(model, x[1:n] >= 0)
  @variable(model, vA)
  @objective(model, Max, vA)
  @constraint(model, obj[j = 1:n], vA <= sum(x[i] * A[i, j] for i = 1:m))
  @constraint(model, simplex, sum(x) == 1)
  optimize!(model)
  solution_summary(model)
  return objective_value(model), value.(x)
end

function maxmin_value_B(G::Game)
  B = G.B
  m, n = size(A)
  model = Model(GLPK.Optimizer)
  @variable(model, y[1:n] >= 0)
  @variable(model, vB)
  @objective(model, Max, vB)
  @constraint(model, obj[i = 1:m], vB <= sum(B[i, j] * y[j] for j = 1:n))
  @constraint(model, simplex, sum(y) == 1)
  optimize!(model)
  solution_summary(model)
  return objective_value(model), value.(y)
end

##
## best response
##

##
## safety value
##

function safety_value(G::Game, tol=1e-5)
  svA, pi_svA = maxmin_value_A(G)
  svB, pi_svB = maxmin_value_B(G)
  vA, pi_vB = minmax_value_A(G)
  vB, pi_vA = minmax_value_B(G)
  out = Dict()
  out[:svA] = svA
  out[:svB] = svB
  out[:vA] = vA
  out[:vB] = vB
  out[:pi_svA] = pi_svA
  out[:pi_svB] = pi_svB
  out[:pi_vB] = pi_vB
  out[:pi_vA] = pi_vA
  @assert(abs(vA - svA) <= tol)
  @assert(abs(vB - svB) <= tol)
  return out
end

function advantage_game(G::Game)
  sv = safety_value(G)
  A = G.A .- sv[:vA]
  B = G.B .- sv[:vB]
  Rplus = Game(A, B, true)
  return Rplus
end

##
## ebs
##

function lexico_minmax_geq(x, y)
  """
  x ≫ y  <==>  ( x[Lx][1] >= y[Lx][1] ) ∨ ( x[Lx][1] == y[Lx][1] ∧ x[Lx][2] == y[Lx][2] ) 
  """
  @assert(size(x, 1) == 2)
  @assert(size(y, 1) == 2)
  Lx = sortperm(x, rev = true)
  return (x[Lx][1] >= y[Lx][1]) || (x[Lx][1] == y[Lx][1] && x[Lx][2] == y[Lx][2])
end
≫(x::Array, y::Array) = lexico_minmax_geq(x, y)

function ebs_weight(Rplus::Game, a::CartesianIndex, aprime::CartesianIndex)
  @assert(Rplus.adv == true)
  if Rplus.A[a] <= Rplus.B[a] && Rplus.A[aprime] <= Rplus.B[aprime]
    w = 0
  elseif Rplus.A[a] >= Rplus.B[a] && Rplus.A[aprime] >= Rplus.B[aprime]
    w = 1
  else
    num = (Rplus.B[aprime] - Rplus.A[aprime])
    den = (Rplus.A[a] - Rplus.A[aprime]) + (Rplus.B[aprime] - Rplus.B[a])
    w = num / den
  end
  return w
end

function score(Rplus::Game, a::CartesianIndex, aprime::CartesianIndex)
  @assert(Rplus.adv == true)
  w = ebs_weight(Rplus, a, aprime)
  return min(w * Rplus.A[a] + (1 - w) * Rplus.A[aprime],
    w * Rplus.B[a] + (1 - w) * Rplus.B[aprime])
end

function ebs(G::Game)
  @assert(size(G.B) == size(G.A))
  m, n = size(G.A)
  if G.adv == false
    Rplus = advantage_game(G)
  else
    Rplus = G
  end

  # joint actions
  as = []
  for aa = 1:m
    for bb = 1:n
      push!(as, CartesianIndex(aa, bb))
    end
  end

  # 2-sequence of joint actions
  a2s = []
  for a in as
    for aprime in as
      push!(a2s, [a, aprime])
    end
  end

  # 2-sequence action scores
  scores = zeros(length(a2s))
  for (idx, a2) in enumerate(a2s)
    a = a2[1]
    aprime = a2[2]
    scores[idx] = score(Rplus, a, aprime)
  end

  egidx = argmax(scores)
  aeg, aegprime = a2s[egidx]
  w = ebs_weight(Rplus, aeg, aegprime)
  EBS_A = w * G.A[aeg] + (1 - w) * G.A[aegprime]
  EBS_B = w * G.B[aeg] + (1 - w) * G.B[aegprime]
  policy = CorrelatedPolicy(aeg, aegprime, w)
  value = (A = EBS_A, B = EBS_B)
  return policy, value
end

function get_joint_actions(G)
  m, n = size(G.A)
  # joint actions
  jas = []
  for aa = 1:m
    for bb = 1:n
      push!(jas, CartesianIndex(bb, aa)) ## !NOTE! JR updated 2022.01.19
    end
  end
  return jas
end

function get_joint_action_pairs(G)
  jas = get_joint_actions(G)
  # joint action pairs
  japs = []
  for a in jas
    for aprime in jas
      push!(japs, [a, aprime])
    end
  end
  return japs
end

function get_jamap(G)
  jas = get_joint_actions(G)
  idx = collect(1:length(jas))
  return hcat(jas, idx)
end

function ja2u(ja, jamap)
  return float.([Tuple(ja) == x for x in Tuple.(jamap[:, 1])])
end
function u2ja(U, jamap)
  idx = findfirst(U .!= 0)
  return jamap[idx, 1]
end


# function get_jas2idx_map(G)
#   jas = get_joint_actions(G)
#   idx = collect(1:length(jas))
#   return Dict(zip(jas, idx))
# end

# function get_idx2jas_map(G)
#   jas = get_joint_actions(G)
#   idx = collect(1:length(jas))
#   return Dict(zip(idx, jas))
# end