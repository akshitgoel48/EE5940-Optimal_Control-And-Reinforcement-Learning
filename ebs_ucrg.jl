##
## plausible games
##

function get_Ck(N::Matrix, delta::Real)
  return sqrt.(2log(1/delta) ./ N)
end
function get_Ck(N::Matrix, delta::Real, policy::CorrelatedPolicy)
  w = policy.w
  a = policy.a
  aprime = policy.aprime
  return w * sqrt.(2log(1/delta) / N[a]) + (1-w) * sqrt.(2log(1/delta) / N[aprime])
end
function get_plausible_games(Gbar::Game, N::Matrix, delta::Real)
  rbarA = Gbar.A
  rbarB = Gbar.B
  @assert(size(rbarA) == size(rbarB))
  rupA = zeros(size(rbarA))
  rupB = zeros(size(rbarA))
  rdnA = zeros(size(rbarA))
  rdnB = zeros(size(rbarA))

  Ck = get_Ck(N, delta)
  rupA = rbarA .+ Ck
  rupB = rbarB .+ Ck
  rdnA = rbarA .- Ck
  rdnB = rbarB .- Ck

  Gup = Game(rupA, rupB, false)
  Gdn = Game(rdnA, rdnB, false)
  return Gup, Gdn
end

##
## optimistic ebs
##
#=
Here we need to find `SV_dn`` s.t.
$$
SV^i - epsilon_k <= SV_dn^i <= SV^i + epsilon_k, for i = A,B
$$
where SV^i is the maxmin value of player i in the _true_ game
=#
function optimistic_maxmin(Gbar, Gup, Gdn)
  vA, maxmin_policy_A = maxmin_value_A(Gup)
  vB, maxmin_policy_B = maxmin_value_B(Gup)
  br_B = 
  br_A = 
end