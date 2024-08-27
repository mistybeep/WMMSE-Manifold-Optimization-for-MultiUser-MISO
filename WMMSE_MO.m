clc;clear;
K = 1; % 基站个数，此版本固定为1
Nt = 64; % 发射天线个数
Nrf = 16; % 射频链
Nr = 1; % 接收天线个数
epsilon = 1e-3; % 收敛条件
snr = 10; % 信噪比
P = 1; % 发射功率
sigma2 = P/db2pow(snr);% 噪声功率

I = 4; % 用户个数
alpha1 = ones(I,K); % 权重系数，都假设相同

d = 1; % 假设每个用户都有d条路独立的数据流

max_iter = 100;
tic;
% 初始化信道向量
% H = zeros(Nr,Nt,I); % 信道系数 用户数量*每个用户天线数量*基站天线数量
% for i=1:I
%     H(: , :, i)=sqrt(1/2)*(randn(Nr,Nt)+1i*randn(Nr,Nt)); % 圆复高斯信道
% end
load H.mat

rate = []; % 初始化一个空向量记录rate

% 初始化W和U矩阵
U =randn(Nr,d,I) + 1j*randn(Nr,d,I);
W = zeros(d,d,I);
for i=1:I
    W(:,:,i)=eye(d,d);
end

% 初始化波束赋形矩阵
V_RF = exp(1j*unifrnd(0,2*pi,Nt,Nrf));
V_D = zeros(Nrf,d,I); % 每个基站的波束赋形矩阵
for i=1:I
    v = sqrt(1/2)*(randn(Nrf,d)+1i*randn(Nrf,d));
    V_D(:,:, i) = sqrt(P/(I*trace(V_RF*v*v'*V_RF')))*v;
end

V = zeros(Nt,d,I);
for i=1:I
    V(:,:, i) = V_RF*V_D(:,:, i);
end

rate_old = sum_rate(H,V_D,V_RF,sigma2,Nr,I,alpha1);
rate = [rate rate_old];

vrf_manifold = complexcirclefactory(Nt*Nrf);

iter1 = 1;
while(1)
    U = find_U(H, V_D, V_RF, sigma2, P, Nt, Nr, I, d); 
    W = find_W(U, H, V_D, V_RF, Nt, Nr, I, d, P, sigma2); 
    V_RF = WMMSE_MO_Vrf_algorithm(V_RF, V_D, H, W, U, alpha1, sigma2, P,vrf_manifold);
    [V_D, P_temp] = find_V(alpha1,V_RF,H,sigma2,U,W,Nrf, Nr, I,d ,P); 
    for i=1:I
        V_D(:,:, i) = sqrt(P/(P_temp))*V_D(:,:,i);
        V(:,:, i) = V_RF*V_D(:,:, i);
    end
    rate_new = sum_rate(H,V_D,V_RF,sigma2,Nr,I,alpha1); % 计算和速率
    rate = [rate rate_new];
    iter1 = iter1 + 1;
    if abs(rate_new-rate_old) / rate_old < epsilon || iter1 > max_iter
        break;
    end
    rate_old = rate_new;
end
a=0;
for i=1:I
    a = a+trace(V_RF*V_D(:,:,i)*V_D(:,:,i)'*V_RF');
end
toc;
plot(0:iter1-1,rate,'r-o')
grid on
xlabel('Iterations')
ylabel('Sum rate (bits per channel use)')
set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1)
title('WMMSE, K=1, T=128, R=4, d=4, 10dB, \epsilon=1e-3','Interpreter','tex')