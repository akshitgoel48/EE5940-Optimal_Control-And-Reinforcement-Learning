using LinearAlgebra

"""
assumes that P_A == P_B
"""
function thompson_update(u::Vector, XA::Real, XB::Real, MU_A::Matrix, MU_B::Matrix, P::AbstractArray, sigma::Real)
  # setup
  mu_A = vec(MU_A)
  mu_B = vec(MU_B)

  # update means
  num_A = XA - u' * mu_A
  num_B = XB - u' * mu_B
  den = (u' * P * u) + sigma^2
  mu_A = mu_A + (num_A / den) * P * u
  mu_B = mu_B + (num_B / den) * P * u

  # update covariances
  num = (P * u) * (u' * P)
  P = P - num ./ den

  return reshape(mu_A, size(MU_A)), reshape(mu_B, size(MU_B)), P
end

function thompson_sample(MU_A, MU_B, P)
  L = cholesky(P).L
  A = vec(MU_A) + L * randn(prod(size(MU_A)))
  B = vec(MU_B) + L * randn(prod(size(MU_A)))
  A = reshape(A, size(MU_A))
  B = reshape(B, size(MU_B))
  G = Game(A, B, false)
  return G
end

function thompson_rewards(u::Vector, thetastar_A::Vector, thetastar_B::Vector, sigma::Real)
  XA = u' * thetastar_A + sigma*randn()
  XB = u' * thetastar_B + sigma*randn()
  return XA, XB
end

function run_thompson_sampling(
  MU_A0::Matrix, MU_B0::Matrix, P0::AbstractArray, sigma0::Real,
  Gtrue::Game,
  totalrounds=1_000, batchsize=1, tol=1e-5
  )

  # initialize
  jamap = get_jamap(Gtrue)
  thetastar_A = vec(Gtrue.A)
  thetastar_B = vec(Gtrue.B)
  MU_A, MU_B, P, sigma = MU_A0, MU_B0, P0, sigma0
  actions = Array{CartesianIndex}(undef, totalrounds)
  posterior_means = Array{Game}(undef, totalrounds)
  posterior_ebs = Array{NamedTuple}(undef, totalrounds)
  rewards = zeros(2, totalrounds)
  N = zeros(size(Gtrue.A))
  empirical_freqs = zeros(size(Gtrue.A)..., totalrounds)
  outerrounds = Int(totalrounds/batchsize)

  r = 0 # rounds played
  # run
  for k in 1:outerrounds
    # sample a game
    Gs = thompson_sample(MU_A, MU_B, P)

    # compute EBS
    policy_eg, value_eg = ebs(Gs)
    w = policy_eg.w
    
    # play policy
    for t in 1:batchsize
      r += 1
      posterior_ebs[r] = value_eg
      if (tol <= w) && (w <= 1-tol)
        # randomize between a and aprime
        z = rand()
        if z <= w #! double check
          ja = policy_eg.a
        else
          ja = policy_eg.aprime
        end
      
      elseif w <= tol
        # choose single action aprime
        ja = policy_eg.aprime
      
      elseif w >= 1-tol
        # choose single action a
        ja = policy_eg.a
      end
      u = ja2u(ja, jamap)
      actions[r] = ja
      N[ja] += 1
      empirical_freqs[:,:,r] = N ./ r

      # get reward
      XA, XB = thompson_rewards(u, thetastar_A, thetastar_B, sigma)
      rewards[1,r] = XA
      rewards[2,r] = XB

      # update beliefs
      MU_A, MU_B, P = thompson_update(u, XA, XB, MU_A, MU_B, P, sigma)
      posterior_means[r] = Game(MU_A, MU_B, false)
    end
  end

  out = Dict()
  out[:actions] = actions
  out[:rewards] = rewards
  out[:posterior_means] = posterior_means
  out[:posterior_ebs] = posterior_ebs
  out[:empirical_freqs] = empirical_freqs
  out[:N] = N
  out[:MU_A] = MU_A
  out[:MU_B] = MU_B
  out[:P] = P

  return out
end

function run_modified_thompson_sampling(
  MU_A0::Matrix, MU_B0::Matrix, P0::AbstractArray, sigma0::Real,
  Gtrue::Game,
  totalrounds=1_000, batchsize=1, tol=1e-5
  )

  # initialize
  jamap = get_jamap(Gtrue)
  thetastar_A = vec(Gtrue.A)
  thetastar_B = vec(Gtrue.B)
  MU_A, MU_B, P, sigma = MU_A0, MU_B0, P0, sigma0
  actions = Array{CartesianIndex}(undef, totalrounds)
  posterior_means = Array{Game}(undef, totalrounds)
  posterior_ebs = Array{NamedTuple}(undef, totalrounds)
  rewards = zeros(2, totalrounds)
  N = zeros(size(Gtrue.A))
  empirical_freqs = zeros(size(Gtrue.A)..., totalrounds)
  outerrounds = Int(totalrounds/batchsize)

  r = 0 # rounds played
  # run
  for k in 1:outerrounds
    # sample a game
    Gs = thompson_sample(MU_A, MU_B, P)

    # compute EBS
    policy_eg, value_eg = ebs(Gs)
    w = policy_eg.w
    
    # play policy
    for t in 1:batchsize
      r += 1
      posterior_ebs[r] = value_eg
      
      # find action that makes the empirical frequency most aligned with estimated policy_eg
      # if r > 1
      #   ef = empirical_freqs[:,:,r-1]
      # else
      #   ef = empirical_freqs[:,:,r]
      # end
      ef = N ./ r
      pf = zeros(2,2)
      pf[policy_eg.a] = policy_eg.w
      pf[policy_eg.aprime] = 1-policy_eg.w
      errs = Inf*ones(4)
      acts = Array{CartesianIndex}(undef,4)
      for ja in jamap[:,1]
        N_proposed = deepcopy(N)
        N_proposed[ja] += 1
        ef = N_proposed./sum(N_proposed)
        u = ja2u(ja, jamap)
        idx = findfirst(u .== 1)
        errs[idx] = norm(ef - pf)
      end
      ja = jamap[argmin(errs),1]
      u = ja2u(ja, jamap)
      actions[r] = ja
      N[ja] += 1
      empirical_freqs[:,:,r] = N ./ r

      # get reward
      XA, XB = thompson_rewards(u, thetastar_A, thetastar_B, sigma)
      rewards[1,r] = XA
      rewards[2,r] = XB

      # update beliefs
      MU_A, MU_B, P = thompson_update(u, XA, XB, MU_A, MU_B, P, sigma)
      posterior_means[r] = Game(MU_A, MU_B, false)
    end
  end

  out = Dict()
  out[:actions] = actions
  out[:rewards] = rewards
  out[:posterior_means] = posterior_means
  out[:posterior_ebs] = posterior_ebs
  out[:empirical_freqs] = empirical_freqs
  out[:N] = N
  out[:MU_A] = MU_A
  out[:MU_B] = MU_B
  out[:P] = P

  return out
end


# function run_thompson_sampling(
#   MU_A0::Matrix, MU_B0::Matrix, P0::AbstractArray, sigma0::Real,
#   Gtrue::Game,
#   maxiter=1_000, batch=1, tol=1e-5
#   )

#   # initialize
#   jamap = get_jamap(Gtrue)
#   thetastar_A = vec(Gtrue.A)
#   thetastar_B = vec(Gtrue.B)
#   MU_A, MU_B, P, sigma = MU_A0, MU_B0, P0, sigma0
#   total_rounds = maxiter * batch
#   rewards = zeros(2, total_rounds)
#   actions = Array{CartesianIndex}(undef, total_rounds)
#   posterior_means = Array{Game}(undef, total_rounds)
#   posterior_ebs = Array{NamedTuple}(undef, total_rounds)
#   N = zeros(size(G.A))
#   empirical_freqs = zeros(size(G.A)..., total_rounds)

#   r = 0 # rounds played
#   # run
#   while true
#     if r > total_rounds
#       break
#     end

#     # sample a game
#     Gs = thompson_sample(MU_A, MU_B, P)

#     # compute EBS
#     policy_eg, value_eg = ebs(Gs)
#     w = policy_eg.w
    
#     # play policy
#     t = 0
#     while true
#       r += 1
#       t += 1
#       posterior_ebs[r] = value_eg
#       println("r=", r)
#       println("t=", t)
#       if t > batch
#         break
#       end
#       if (tol <= w) && (w <= 1-tol)
#         # randomize between a and aprime
#         z = rand()
#         if z <= w #! double check
#           ja = policy_eg.a
#         else
#           ja = policy_eg.aprime
#         end
      
#       elseif w <= tol
#         # choose single action aprime
#         ja = policy_eg.aprime
      
#       elseif w >= 1-tol
#         # choose single action a
#         ja = policy_eg.a
#       end
#       u = ja2u(ja, jamap)
#       actions[r] = ja
#       N[ja] += 1
#       empirical_freqs[:,:,r] = N ./ r

#       # get reward
#       println("u=",u)
#       XA, XB = thompson_rewards(u, thetastar_A, thetastar_B, sigma)
#       println("XA=",XA)
#       println("XB=",XB)
#       println()
#       rewards[1,r] = XA
#       rewards[2,r] = XB

#       # update beliefs
#       MU_A, MU_B, P = thompson_update(u, XA, XB, MU_A, MU_B, P, sigma)
#       posterior_means[r] = Game(MU_A, MU_B, false)
#     end
#   end

#   out = Dict()
#   out[:actions] = actions
#   out[:rewards] = rewards
#   out[:posterior_means] = posterior_means
#   out[:posterior_ebs] = posterior_ebs
#   out[:empirical_freqs] = empirical_freqs
#   out[:N] = N
#   out[:MU_A] = MU_A
#   out[:MU_B] = MU_B
#   out[:P] = P

#   return out
# end