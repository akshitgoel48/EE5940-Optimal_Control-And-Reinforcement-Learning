using Distributed
addprocs(10)
using Random
Random.seed!(123456)

@everywhere begin
import Pkg;
Pkg.activate(".");
Pkg.instantiate();
include("ebs.jl")
include("ebs_thompson.jl")
using Random
using Printf
using DelimitedFiles
using Distributed

##
## true game
##

A = [4/5 1/10; 9/5 3/10];
B = [4/5 9/5; 0 3/10];
Gtrue = Game(A, B, false)
policy_eg, value_eg = ebs(Gtrue)
G=deepcopy(Gtrue)


##
## parameters
##

#=
Note: here we view the reward for player $i=A,B$ in round $t$ as
$$
X_t^{(i)} = u_t^\top {\theta^{(i)}}^* + W_t
$$
where $u_t$ is the joint action in round $t$ and $W_t$ is the noise
where $W_t \sim N(0,\sigma^2)$.

Note: the EBS _value_ (is) unique but the _policy_ is NOT
=#

# prior distribution on rewards
MU_A0 = zeros(size(Gtrue.A))
MU_B0 = zeros(size(Gtrue.B))
P0 = 0.1Diagonal(ones(length(MU_A0)))
sigma0 = 0.25 # std of action reward, $W_t$

# run parameters
totalrounds = 2^14
tol = 1e-5
end# everywhere

#=
#
# simulations
#
include("ebs_thompson.jl")
# fig, ax = subplots(figsize=(4,4))
ax.cla()

batches = [1,2,4,8,16]#,16,32,64,128]
batches = [64,128]
for i in eachindex(batches)
  # include("ebs_thompson.jl")
  batchsize = batches[i]
  # # run thompson sampling
  out = run_thompson_sampling(MU_A0, MU_B0, P0, sigma0, Gtrue, totalrounds, batchsize, tol);
  batch_str = string(batchsize)
  sigma_str = replace(@sprintf("%0.2f", sigma0), "."=>",")
  P0_str    = replace(@sprintf("%0.2f", P0[1,1]), "."=>",")
  out[:fname] = "figures/$(batch_str)__$(sigma_str)__$(P0_str)"

  # plot output
  println("plotting: $(out[:fname])")
  plot_output(out, Gtrue, fig, ax)

  # run modified thompson sampling
  include("ebs_thompson.jl")
  out = run_modified_thompson_sampling(MU_A0, MU_B0, P0, sigma0, Gtrue, totalrounds, batchsize, tol);
  batch_str = string(batchsize)
  sigma_str = replace(@sprintf("%0.2f", sigma0), "."=>",")
  P0_str    = replace(@sprintf("%0.2f", P0[1,1]), "."=>",")
  out[:fname] = "figures/empirical__$(batch_str)__$(sigma_str)__$(P0_str)"
  
  # plot output
  println("plotting: $(out[:fname])")
  plot_output(out, Gtrue, fig, ax)

end
=#

#
# comparison to ucrg
#

@everywhere begin
nrun = 50
function simtsr!(i)
  println("run = $i")
  out = run_thompson_sampling(deepcopy(MU_A0), deepcopy(MU_B0), deepcopy(P0), deepcopy(sigma0), deepcopy(Gtrue), 100_000, 1, tol);
  actions = out[:actions];
  rewards = out[:rewards];
  posterior_means = out[:posterior_means];
  posterior_ebs = out[:posterior_ebs];
  empirical_freqs = out[:empirical_freqs];
  N = out[:N];
  MU_A = out[:MU_A];
  MU_B = out[:MU_B];
  P = out[:P];

  Gbelief = Game(MU_A, MU_B, false)
  policy_eg_belief, value_eg_belief = ebs(Gbelief)
  policy_eg_true, value_eg_true = ebs(Gtrue)
  totalrounds = size(rewards,2)

  ebs_rewards = [value_eg_true.A; value_eg_true.B]' .* hcat(collect(1:size(rewards,2)), collect(1:size(rewards,2)))
  ts_rewards = cumsum(rewards', dims=1)
  regret = ebs_rewards - ts_rewards
  writedlm("output/regret_tsr_$i.csv", regret)
end
end
pmap(i->simtsr!(i), 1:nrun)
# out[:fname] = "figures/tsr100k"
# plot_output(out, Gtrue, fig, ax)

@everywhere begin
nrun = 50
function simtse!(i)
  println("run = $i")
  out = run_modified_thompson_sampling(deepcopy(MU_A0), deepcopy(MU_B0), deepcopy(P0), deepcopy(sigma0), Gtrue, 100_000, 1, tol);
  actions = out[:actions];
  rewards = out[:rewards];
  posterior_means = out[:posterior_means];
  posterior_ebs = out[:posterior_ebs];
  empirical_freqs = out[:empirical_freqs];
  N = out[:N];
  MU_A = out[:MU_A];
  MU_B = out[:MU_B];
  P = out[:P];

  Gbelief = Game(MU_A, MU_B, false)
  policy_eg_belief, value_eg_belief = ebs(Gbelief)
  policy_eg_true, value_eg_true = ebs(Gtrue)
  totalrounds = size(rewards,2)

  ebs_rewards = [value_eg_true.A; value_eg_true.B]' .* hcat(collect(1:size(rewards,2)), collect(1:size(rewards,2)))
  ts_rewards = cumsum(rewards', dims=1)
  regret = ebs_rewards - ts_rewards
  writedlm("regret_tse_$i.csv", regret)
end
end# everywhere
pmap(i->simtse!(i), 1:nrun)
# out[:fname] = "figures/tse100k"
# plot_output(out, Gtrue, fig, ax)
  

#=
Notes:
- some actions never get player (i.e., (2,2)) so we never estimate those rewards
- increasing sigma0 seems to help rule against the bias toward better rewards for A
=#

