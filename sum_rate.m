function system_rate = sum_rate(H,V_D,V_RF,sigma2,R,I,alpha1)
    rate = zeros(I,1);
    for i=1:I
            denominator = zeros(R,R);
            for l=1:I
                denominator = denominator + H(:,:,i)*V_RF*V_D(:,:,l)*V_D(:,:,l)'*V_RF'*H(:,:,i)';
            end
            numerator = H(:,:,i)*V_RF*V_D(:,:,i)*V_D(:,:,i)'*V_RF'*H(:,:,i)';
            denominator = denominator - numerator + sigma2*eye(R);

            rate(i) = log2(det(eye(R)+numerator / denominator));
    end
    system_rate = real(sum(rate.*alpha1,'all'));
end