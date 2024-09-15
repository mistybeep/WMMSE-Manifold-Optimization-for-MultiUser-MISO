import random
import scipy.io as sio
import torch
import numpy as np
import math
import time

from net import *  # 假设你有这些文件

# 设置随机种子
seed = 42  # 固定随机种子
torch.manual_seed(seed)  # CPU随机种子
np.random.seed(seed)  # numpy随机种子
random.seed(seed)  # python随机种子
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 加载数据，确保数据加载到CPU
HH = sio.loadmat(f'dataset.mat')['HH']  # 加载信道H，numpy格式
V_RF = sio.loadmat(f'dataset.mat')['V_RF_train']
V_D = sio.loadmat(f'dataset.mat')['V_D_train']
user_weights = sio.loadmat(f'dataset.mat')['omega'].squeeze()  # 加载用户权重
regulated_user_weights = user_weights / np.sum(user_weights)

# 将numpy数组转换为torch张量并确保它们在CPU上
H = torch.tensor(HH).to('cpu')
V_RF = torch.tensor(V_RF).to('cpu')
V_D = torch.tensor(V_D).to('cpu')

# 训练过程
variable_name = f"WSR_List_MLMO_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}"
file_name = f'./WSR_List_MLMO_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}.mat'
globals()[variable_name] = torch.zeros(nr_of_training, External_iteration)  # 记录每个样本的WSR

MLMO_run_time = 0

for item_index in range(nr_of_training):
    optimizer_vd = MetaOptimizerVd(input_size_vd, hidden_size_vd, output_size_vd)
    adam_vd = torch.optim.Adam(optimizer_vd.parameters(), lr=optimizer_lr_vd)
    optimizer_vrf = MetaOptimizerVrf(input_size_vrf, hidden_size_vrf, output_size_vrf)
    adam_vrf = torch.optim.Adam(optimizer_vrf.parameters(), lr=optimizer_lr_vrf)

    maxi = 0
    mm_Wave_Channel = H[item_index, :, :].to(torch.complex64).to('cpu')  # 确保在CPU上
    Vrf = V_RF[item_index, :, :].to(torch.complex64).to('cpu')  # 确保在CPU上
    Vrf_init = Vrf
    Vd = V_D[item_index, :, :, :].to(torch.complex64).to('cpu')  # 确保在CPU上
    Vd = Vd.squeeze(1)
    transmitter_precoder_init = Vd
    transmitter_precoder = transmitter_precoder_init

    LossAccumulated_vd = 0  # 记录预编码矩阵元学习网络的累积损失
    LossAccumulated_vrf = 0  # 记录相移矩阵元学习网络的累积损失
    start_time = time.time()

    for epoch in range(External_iteration):
        loss_vrf, sum_loss_vrf, Vrf = meta_learner_vrf(optimizer_vrf, 2,
                                                       regulated_user_weights, mm_Wave_Channel,
                                                       Vd.clone().detach(),
                                                       Vrf_init)

        loss_w, sum_loss_w, Vd = meta_learner_vd(optimizer_vd, Internal_iteration,
                                                 regulated_user_weights, mm_Wave_Channel,
                                                 transmitter_precoder_init,
                                                 Vrf.clone().detach())

        transmitter_precoder = Vd
        normV = torch.norm(Vrf @ transmitter_precoder)  # 计算归一化前的预编码矩阵范数
        WW = math.sqrt(total_power) / normV
        transmitter_precoder = transmitter_precoder * WW

        loss_total = -compute_weighted_sum_rate(mm_Wave_Channel, transmitter_precoder, Vrf, regulated_user_weights)
        LossAccumulated_vd += loss_total
        LossAccumulated_vrf += loss_total
        WSR = compute_weighted_sum_rate(mm_Wave_Channel, transmitter_precoder, Vrf.detach(), regulated_user_weights)
        globals()[variable_name][item_index, epoch] = WSR

        if WSR > maxi:
            maxi = WSR.item()
            print('max', maxi, 'epoch', epoch, 'item', item_index)

        if (epoch + 1) % Update_steps == 0:
            adam_vd.zero_grad()
            adam_vrf.zero_grad()
            Average_loss_vd = LossAccumulated_vd / Update_steps
            Average_loss_vrf = LossAccumulated_vrf / Update_steps
            Average_loss_vd.backward(retain_graph=True)
            Average_loss_vrf.backward(retain_graph=True)
            adam_vd.step()
            adam_vrf.step()

            WSR = compute_weighted_sum_rate(mm_Wave_Channel, transmitter_precoder, Vrf.detach(), regulated_user_weights)
            LossAccumulated_vd = 0
            LossAccumulated_vrf = 0

    end_time = time.time()
    MLMO_run_time += end_time - start_time
    print("时间：", end_time - start_time)

# 计算平均运行时间
average_MLMO_run_time = MLMO_run_time / nr_of_training

# 保存.mat文件
data_to_save = {
    variable_name: globals()[variable_name].detach().numpy(),
    'average_MLMO_run_time': average_MLMO_run_time
}
sio.savemat(file_name, data_to_save)
