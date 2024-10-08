function [V,P_temp] = find_V(alpha1, V_RF, H, sigma2, U, W, T , R ,I ,d ,P )

    J=zeros(T, T);
    V=zeros(T,d, I);
   
    for l=1:I
        J = J + alpha1(l, 1) *V_RF'*H(:,:,l)'*U(:,:,l)*W(:,:,l)*(U(:,:,l)')*(H(:,:,l)*V_RF) + ... %
            eye(T, T)*sigma2*real(trace(alpha1(l,1)*U(:,:,l)*W(:,:,l)*(U(:,:,l)')))/P ;
    end

    % max_iter = 100; % 二分法查找最优对偶变量\mu
    % mu = zeros(1,1);
    % mu_min = 0;
    % mu_max = 10;
    % iter = 0;
    % while(1)
    %     mu1 = (mu_max+mu_min) / 2;
    %     P_tem = 0;
    %
    %     for i=1:I % 计算功率和
    %         V_tem = ((J+mu1*eye(T))) \ (alpha1(i,1)*(H(:,:,i)'*U(:,:,i)*W(:,:,i)));
    %         P_tem = P_tem + real(trace(V_tem*V_tem'));
    %     end
    %
    %     if P_tem > P
    %         mu_min = mu1;
    %     else
    %         mu_max = mu1;
    %     end
    %     iter = iter + 1;
    %
    %     if abs(mu_max - mu_min) < 1e-5 || iter > max_iter
    %         break
    %     end
    %
    % end
    %
    % mu = mu1;
    P_temp = 0;

    for l=1:I
        V(:,:,l) = J \ (alpha1(l, 1) * (V_RF'*H(:,:,l)'*U(:,:,l)*W(:,:,l)));
        P_temp  = P_temp + trace(V_RF*V(:,:,l)*V(:,:,l)'*V_RF');
    end

end