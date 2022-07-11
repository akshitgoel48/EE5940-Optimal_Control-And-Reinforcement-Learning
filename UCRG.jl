# A : joint action space
# N_k(a) : Num rounds action played in epsiode k
# N_k : Num rounds episode k lasted
# N_tk(a) : Num rounds action played upto round tk
# rbar^i_k(a) : Empirical avg. rewards of player i for actiona upto round tk

m = 2;  n = 2;

# joint actions
A = []
for aa = 1:m
  for bb = 1:n
    push!(A, CartesianIndex(aa, bb))
  end
end

ϵ_k  = 0.01;    δ_k = 0.01

t = 1
N_k = 0; Naction_k = Dict(); N_tk = Dict()

for a in A
    N_k[a] = 0
    N_tk[a] = 0
end

num_episodes = 100
for k = 1:num_episodes
    t_k = t
    for i = 1:2
        for a  in A
            r_uhat_k[i,a] = r_bar_k[i,a] + C_k[a]
            r_lhat_k[i,a] = r_bar_k[i,a] - C_k[a]
        end
    end
   pi_tilde_k = OptEgalitarianPolicy(r_bar_k, r_uhat_k, r_lhat_k, ϵ_k)

   # Execute policy pi_tilde_k
   thrshld = max{1, N_tk}
    while true
        a_t = PLAY(pi_tilde_k)
        # r_t =
        N_k = N_k + 1
        Naction_k[a_t] = Naction_k[a_t] + 1
        Naction_tk[a_t] = Naction_tk[a_t] + 1
        for i = 1:2
            r_bar_k[i,a_t] = (r_bar_k[i,a_t]*(N_tk[a_t]-1) + r_t[i])/Naction_tk[a_t]
        end
        if Naction_k[a_t] > thrshld
            break;
        end
    end
end


function PLAY(policy)
    a_t = argmin{policy-Naction_k/N_k}
    if size(a_t) > 1
        # break ties;
    end
    return a_t
end

function bonus(policy)
    C_k = 0;
    constant = sqrt(log(1/δ_k)/1.99)
    if size(policy) == 1
        C_k = constant/sqrt(N_tk[policy])
    else    
        for a in A
            C_k = sum + policy[a]*constant/sqrt(N_tk[a])
        end
    end
    return C_k
end

function argmax_actions(policy)
    tmp = -Inf
    a_argmax = NaN
    for a in A            
        if 2*bonus(a) > ϵ && policy[a] > tmp
            tmp = policy_hat_Eg[a]
            a_argmax = a
        end
    end    
    return a_argmax
end

function OptMaximin(r_bar, r_uhat, r_lhat, i)    
    pi_uhat_SV = 
    pi_lhat_SV_uhat = 
    SV_lhat =     
    return pi_uhat_SV, pi_lhat_SV_uhat, SV_lhat
end

function OptEgalitarianPolicy(r_bar, r_uhat, r_lhat, ϵ)
    for i = 1:2
        pi_uhat_SV[i], pi_lhat_SV_uhat[-i], SV_lhat[i] = OptMaximin(r_bar, r_uhat, r_lhat, i)
        for a in A
            r_uhat_plus[i,a] = r_uhat[i,a] - SV_lhat[i]
        end
    end
    V_uhat, pi_hat_Eg = EBSpolicy(r_uhat_plus)
    pi_hat = pi_hat_Eg
    V_uhat_plus =  V_uhat - SV_lhat

    calA = {}
    for i = 1:2
        for a in A
            if r_uhat_plus[i,a] + ϵ >= V_uhat_plus[i] && r_uhat[i,a] >= 0
                calA[i] = calA[i].append(a)
            end 
        end
    end

    P_tilde = {}
    for i = 1:2
        a_hat[i] = argmax_a ∈ calA[-i] r_uhat_plus[i,a]
        if r_uhat_plus[i,a_hat[i]] > V_uhat_plus[i]
            P_tilde = P_tilde.append(i)
        end
    end

    if size(P_tilde) > 0
        if r_uhat_plus[1,a_hat[1]] >= r_uhat_plus[2,a_hat[2]]
            p_tilde = 1
        else
            p_tilde = 2
        end        
        pi_hat = a_hat[p_tilde]
    end

    if 2*bonus(pi_hat_Eg) > ϵ
        a_hat_Eg = argmax_actions(pi_hat_Eg)
        pi_hat = a_hat_Eg
    end

    for i = 1:2
        pi_hat_SV = pi_uhat_SV[i]*pi_lhat_SV_uhat[-i]
        if 2*bonus(pi_hat_SV) > ϵ
            a_hat_SV = argmax_actions(pi_hat_SV)
            pi_hat = a_hat_SV
    end

    return pi_hat
end