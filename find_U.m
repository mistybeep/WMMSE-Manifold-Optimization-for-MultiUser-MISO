function U = find_U(H, V_D, V_RF, sigma2, P, Nt, Nr, I, d)

    J = zeros(Nr,Nr,I);  %计算不含噪声项的矩阵
    U = zeros(Nr,d,I);

    for i=1:I
        for l=1:I
            J(:,:,i) = J(:,:,i) + H(:,:,i)*V_RF*V_D(:,:,l)*(V_RF*V_D(:,:,l))'*(H(:,:,i)') + ...%
                sigma2*eye(Nr,Nr)*real(trace(V_RF*V_D(:,:,l)*(V_RF*V_D(:,:,l))'))/P;
        end
  
        U(:,:,i) = J(:,:,i) \ (H(:,:,i)*V_RF*V_D(:,:,i));% + sigma2*eye(R,R)) \ (H(:,:,i)*V(:,:,i))
    end     
end