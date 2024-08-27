function egrad = vrf_egrad(x, Nt, Nrf, V_D, H, W, U, alpha1, sigma2, P)
    % 输入参数：
    % x: 预编码矩阵 (N_RF x N_t)
    % V_D: 数字预编码矩阵 (N_RF x K x K) (三维数组，最后一维是用户数)
    % H: 信道矩阵 (N_r x N_t x K) (三维数组，最后一维是用户数)
    % W: 对角矩阵 (N_r x N_r x K) (三维数组，最后一维是用户数)
    % U: 用户矩阵 (N_r x N_r x K) (三维数组，最后一维是用户数)
    % alpha1: 权重向量 (K x 1)
    % sigma2: 噪声功率 (标量)
    % P: 发射功率 (标量)

    V_RF = reshape(x,Nt,Nrf);
    K = size(V_D, 3);  % 用户数

    egrad = zeros(size(V_RF));  % 初始化梯度矩阵

    for k = 1:K
        % 计算 G_tilde_k = H_k^H * U_k
        G_tilde_k = H(:,:,k)' * U(:,:,k);
        
        % 计算 sum_m(V_D_m * V_D_m^H)
        sum_VDm_VDmH = zeros(Nrf, Nrf);
        for m = 1:K
            sum_VDm_VDmH = sum_VDm_VDmH + V_D(:,:,m) * V_D(:,:,m)';
        end
        
        % 计算各项的梯度
        term1 = -alpha1(k) * G_tilde_k * W(:,:,k) * V_D(:,:,k)';
        term2 = alpha1(k) * G_tilde_k * W(:,:,k) * G_tilde_k' * V_RF * sum_VDm_VDmH;
        term3 = alpha1(k) * (sigma2 / P) * trace(U(:,:,k) * W(:,:,k) * U(:,:,k)') * V_RF * sum_VDm_VDmH;
        
        % 累加梯度
        egrad = egrad + term1 + term2 + term3;
    end
    egrad = egrad(:);
end
