using PyPlot

function plot_output(out, Gtrue, fig, ax)
  actions = out[:actions];
  rewards = out[:rewards];
  posterior_means = out[:posterior_means];
  posterior_ebs = out[:posterior_ebs];
  empirical_freqs = out[:empirical_freqs];
  N = out[:N];
  MU_A = out[:MU_A];
  MU_B = out[:MU_B];
  P = out[:P];
  fname = out[:fname];

  Gbelief = Game(MU_A, MU_B, false)
  policy_eg_belief, value_eg_belief = ebs(Gbelief)
  policy_eg_true, value_eg_true = ebs(Gtrue)
  # sum(rewards, dims=2)
  totalrounds = size(rewards,2)

  ebs_rewards = [value_eg_true.A; value_eg_true.B]' .* hcat(collect(1:size(rewards,2)), collect(1:size(rewards,2)))
  ts_rewards = cumsum(rewards', dims=1)
  regret = ebs_rewards - ts_rewards

  #
  # regret plot
  #
  # num_K = 4
  # delta = 0.01
  # ub = [4.007 * ((num_K * log((16 * num_K * log(t) + 2 * num_K + 1) / delta)) ^ (1. / 3)) * (t ^ (2. / 3)) for t in 1:totalrounds]
  ax.cla()
  ax.plot(regret, label=["A";"B"])
  # ax.plot(ub, label="upper bound")
  ax.set_xlabel("rounds played")
  ax.set_ylabel("regret (vs EBS)")
  ax.legend()
  fig.tight_layout()
  fig.savefig(fname*"_regret.pdf")

  #
  # policy diff plot
  #
  a = zeros(2,2,totalrounds)
  b = zeros(2,2,totalrounds)
  for k in 1:totalrounds
    a[:,:,k] = empirical_freqs[:,:,k]
    b[:,:,k] = empirical_freqs[:,:,k]
  end
  plotevery = 100
  ax.cla()
  ax.semilogy(collect(1:plotevery:totalrounds), [a[1,1,k] - 0.0 for k in 1:plotevery:totalrounds], label="(1,1)")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(a[1,2,k] - policy_eg.w) for k in 1:plotevery:totalrounds], label="(1,2)", linestyle=":")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(a[2,1,k] - (1-policy_eg.w)) for k in 1:plotevery:totalrounds], label="(2,1)", linestyle=":")
  ax.semilogy(collect(1:plotevery:totalrounds), [a[2,2,k] - 0.0 for k in 1:plotevery:totalrounds], label="(2,2)")
  ax.legend()
  ax.set_xlabel("rounds played")
  ax.set_ylabel("absolute difference from EBS policy")
  fig.tight_layout()
  fig.savefig(fname*"_ebspolicydiff.pdf")
  
  #
  # estimation diff plots
  #
  a = zeros(2,2,totalrounds)
  b = zeros(2,2,totalrounds)
  for k in 1:totalrounds
    a[:,:,k] = posterior_means[k].A
    b[:,:,k] = posterior_means[k].B
  end
  # A
  ax.cla()
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(a[1,1,k] - Gtrue.A[1,1]) for k in 1:plotevery:totalrounds], label="(1,1)")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(a[1,2,k] - Gtrue.A[1,2]) for k in 1:plotevery:totalrounds], label="(1,2)")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(a[2,1,k] - Gtrue.A[2,1]) for k in 1:plotevery:totalrounds], label="(2,1)")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(a[2,2,k] - Gtrue.A[2,2]) for k in 1:plotevery:totalrounds], label="(2,2)")
  ax.legend()
  ax.set_xlabel("rounds played")
  ax.set_ylabel("absolute difference from game mean")
  fig.tight_layout()
  fig.savefig(fname*"_estimationdiff_A.pdf")
  # B
  ax.cla()
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(b[1,1,k] - Gtrue.B[1,1]) for k in 1:plotevery:totalrounds], label="(1,1)")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(b[1,2,k] - Gtrue.B[1,2]) for k in 1:plotevery:totalrounds], label="(1,2)")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(b[2,1,k] - Gtrue.B[2,1]) for k in 1:plotevery:totalrounds], label="(2,1)")
  ax.semilogy(collect(1:plotevery:totalrounds), [abs(b[2,2,k] - Gtrue.B[2,2]) for k in 1:plotevery:totalrounds], label="(2,2)")
  ax.legend()
  ax.set_xlabel("rounds played")
  ax.set_ylabel("absolute difference from game mean")
  fig.tight_layout()
  fig.savefig(fname*"_estimationdiff_B.pdf")
end