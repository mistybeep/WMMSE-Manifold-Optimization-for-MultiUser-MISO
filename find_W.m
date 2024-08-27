function W = find_W(U, H, V_D, V_RF, Nt, Nr, I, d, P, sigma2)
    W = zeros(d,d,I);
    V = zeros(Nt,d,I);

    for i=1:I
        P_temp = 0;
        J = zeros(d,d);
        % W(:,:,i) = inv(eye(d)-U(:,:,i)'*H(:,:,i)*V(:,:,i)); 
        for j = 1:I
            P_temp = P_temp + trace(V_RF*V_D(:,:,j)*(V_RF*V_D(:,:,j))');
            J = J + U(:,:,i)'*H(:,:,i)*V_RF*V_D(:,:,j)*(V_RF*V_D(:,:,j))'*H(:,:,i)'*U(:,:,i);
        end
        W(:,:,i) = inv(eye(d) - (V_RF*V_D(:,:,i))'*H(:,:,i)'*U(:,:,i) - U(:,:,i)'*H(:,:,i)*V_RF*V_D(:,:,i) + J + P_temp*sigma2*U(:,:,i)'*U(:,:,i)/P);
    end
end