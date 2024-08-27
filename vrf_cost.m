function cost = vrf_cost(x, Nt, Nrf, V_D, H, W, U, alpha1, sigma2, P)
    % 输入参数:
    % x: 射频预编码矩阵 (N_t x N_RF)
    % V_D: 基带预编码矩阵 (N_RF x N_s x K) 
    % H: 信道矩阵 (N_r x N_t x K)
    % W: 权重矩阵 (N_s x N_s x K)
    % U: 奇异矩阵 (N_r x N_s x K)
    % alpha1: 权重向量 (1 x K)

    x = reshape(x,Nt,Nrf);
    K = size(V_D, 3); % 用户数量
    cost = 0; % 初始化目标函数值

    for k = 1:K
        % 计算 G_tilde_k = H_k^H * U_k
        G_tilde_k = H(:, :, k)' * U(:, :, k);

        % 计算目标函数的各项
        term1 = -W(:, :, k) * G_tilde_k' * x * V_D(:, :, k);
        term2 = -W(:, :, k) * V_D(:, :, k)' * x' * G_tilde_k;
        
        term3 = 0;
        term4 = 0;
        for m = 1:K
            term3 = term3 + W(:, :, k) * G_tilde_k' * x * V_D(:, :, m) * V_D(:, :, m)' * x' * G_tilde_k;
            term4 = term4 + trace(x * V_D(:, :, m) * V_D(:, :, m)' * x') * W(:, :, k) * U(:, :, k)' * U(:, :, k);
        end
        
        % 累加各个部分，乘以对应的权重 omega_k
        cost = cost + alpha1(k) * trace(term1 + term2 + term3 + sigma2*term4/P);
    end
end

