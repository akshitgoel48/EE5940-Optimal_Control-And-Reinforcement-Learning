using LinearAlgebra
using PyPlot
using Printf
##
## example 1
##
mutable struct Ex1{T<:Real}
  x   :: Array{T,1}
  y   :: Array{T,1}
  A   :: Array{T,2}
  F   :: SVD
  l1  :: Function
  l2  :: Function
  ξ   :: Function
  H   :: Function
  𝐇   :: Function
  u   :: Function
  v   :: Function
end
function Ex1(A)
  # svd
  F_A = svd(A)
  U,S,V = F_A
  D = Diagonal(S)
  D_12 = sqrt(D)

  m,n = size(A)
  l1(x,y) = x'*A*y
  l2(x,y) = -x'*A*y
  ξ(x,y) = [A*y; -A'*x]
  H(x,y) = [zeros(m,m) A; -A' zeros(n,n)]
  𝐇(x,y) = 0.5(x'*U*D*U'*x + y'*V*D*V'*y)
  u(x) = D_12*U'*x
  v(y) = D_12*V'*y
  xx = randn(m); yy = randn(n)
  @assert(abs(𝐇(xx,yy) - 0.5(sum(u(xx).^2)+sum(v(yy).^2))) <= 1e-8)
  G = Ex1(xx,yy,A,F_A,l1,l2,ξ,H,𝐇,u,v)
  return G
end
m = 3
n = 4
G = Ex1(randn(m,n))

# m = 1
# n = 1
# G = Ex1(randn(m,n))


function integrate(G,parms)
  w = [G.x; G.y]
  history = zeros(length(w),parms[:nsteps]+1)
  l1 = zeros(parms[:nsteps]+1)
  l2 = zeros(parms[:nsteps]+1)
  𝐇 = zeros(parms[:nsteps]+1)
  time = parms[:dt] * collect(0:1:parms[:nsteps])
  history[:,1] = w
  l1[1] = G.l1(w[1:m],w[m+1:end])
  l2[1] = G.l2(w[1:m],w[m+1:end])
  𝐇[1] = G.𝐇(w[1:m],w[m+1:end])
  for i in 2:parms[:nsteps]+1
    w += parms[:dt] * G.ξ(w[1:m],w[m+1:end])
    history[:,i] = w
    l1[i] = G.l1(w[1:m],w[m+1:end])
    l2[i] = G.l2(w[1:m],w[m+1:end])
    𝐇[i] = G.𝐇(w[1:m],w[m+1:end])
  end
  return (time,history,l1,l2,𝐇)
end

fig,ax = subplots(figsize=(10,15),2)
ax1 = ax[1]
ax2 = ax[2]
# ax3 = ax[3]

# parms = Dict()
# parms[:dt] = 1e-3
# parms[:nsteps] = 100_000
# res = integrate(G,parms)
# ax1.cla()
# ax1.plot(res[1][:10:end],res[3][:10:end],label=raw"$V_1$")
# ax1.plot(res[1][:10:end],res[4][:10:end],label=raw"$V_2$")
# ax1.plot(res[1][:10:end],res[5][:10:end],label=raw"$\mathcal{H}$")
# figname = "stepsize = "* @sprintf("%.1e",parms[:dt])
# ax1.set_title(figname)
# ax1.legend()
# ax1.set_xlabel("time")

parms[:dt] = 1e-5
parms[:nsteps] = 1_000_000
res = integrate(G,parms)
ax1.cla()
ax1.plot(res[1][:100:end],res[3][:100:end],label=raw"$V_1$")
ax1.plot(res[1][:100:end],res[4][:100:end],label=raw"$V_2$")
ax1.plot(res[1][:100:end],res[5][:100:end],label=raw"$\mathcal{H}$")
figname = "stepsize = "* @sprintf("%.1e",parms[:dt])
ax1.set_title(figname)
ax1.legend()
ax1.set_xlabel("time")

ax2.cla()
ax2.plot(res[1][:10:100_000],res[3][:10:100_000],label=raw"$V_1$")
ax2.plot(res[1][:10:100_000],res[4][:10:100_000],label=raw"$V_2$")
ax2.plot(res[1][:10:100_000],res[5][:10:100_000],label=raw"$\mathcal{H}$")
figname = "stepsize = "* @sprintf("%.1e",parms[:dt])
ax2.set_title(figname)
ax2.legend()
ax2.set_xlabel("time")

fig.tight_layout()
savefig("/Users/jakeroth/git/5faa124ad2e10a34cb955169/2021-fall/figures/unconstrained_zerosum_bmg_objective.pdf")
# savefig("/Users/jakeroth/git/5faa124ad2e10a34cb955169/2021-fall/figures/unconstrained_1strat_zerosum_bmg_objective.pdf")


parms[:dt] = 1e-5
parms[:nsteps] = 5_000_000
res = integrate(G,parms)

fig,ax = subplots()
ax.cla()
N = 5_000_000
ax.plot(res[1][:100:N],res[2][1,:100:N],label=raw"$\theta_1[1]$")
ax.plot(res[1][:100:N],res[2][m+1,:100:N],label=raw"$\theta_2[1]$")
ax.legend()
figname = "stepsize = "* @sprintf("%.1e",parms[:dt])
ax.set_title(figname)
ax.set_xlabel("time")
fig.tight_layout()
savefig("/Users/jakeroth/git/5faa124ad2e10a34cb955169/2021-fall/figures/unconstrained_zerosum_bmg_state.pdf")
# savefig("/Users/jakeroth/git/5faa124ad2e10a34cb955169/2021-fall/figures/unconstrained_1strat_zerosum_bmg_state.pdf")
