function [V_RF, iter] = WMMSE_MO_Vrf_algorithm(V_RF, V_D, H, W, U, alpha1, sigma2, P,vrf_manifold)

[Nt, Nrf] = size(V_RF);

problem.M = vrf_manifold;

problem.cost = @(x)vrf_cost(x, Nt, Nrf, V_D, H, W, U, alpha1, sigma2, P);
problem.egrad = @(x)vrf_egrad(x, Nt, Nrf, V_D, H, W, U, alpha1, sigma2, P);

L = zeros(Nt,Nrf);
for i = 1 : Nrf %initialize V_RF
    L(:,i) = ones(Nt,1);
end
L = L(:);
[x,iter] = conjugategradient(problem,V_RF(:));

V_RF = reshape(x,Nt,Nrf);