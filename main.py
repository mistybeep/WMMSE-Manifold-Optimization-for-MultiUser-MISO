import random
import scipy.io as sio
import torch

from net import *
import math
import time

import torch.nn.init as init


import psutil
import os


def check_cpu_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")



# </editor-fold>


# <editor-fold desc="set random seed">
seed = 42  # fix the random seed
torch.manual_seed(seed)  # cpu random seed
torch.cuda.manual_seed(seed)  # gpu random seed
torch.cuda.manual_seed_all(seed)  # multi-gpu random seed
np.random.seed(seed)  # numpy random seed
random.seed(seed)  # python random seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# </editor-fold>

HH = sio.loadmat(f'dataset.mat')['HH']  # load the channel H, numpy format
V_RF = sio.loadmat(f'dataset.mat')['V_RF_train']
V_D = sio.loadmat(f'dataset.mat')['V_D_train']
user_weights = sio.loadmat(f'dataset.mat')['omega'].squeeze()  # load the user weights, numpy format
regulated_user_weights = user_weights / np.sum(user_weights)
# mm_Wave_Channel = generate_mmWave_channel_MISO(nr_of_BS_antennas, nr_of_clusters, nr_of_rays, nr_of_users)
H = torch.tensor(HH, dtype=torch.complex64)  # 使用更小的数据类型
V_RF = torch.tensor(V_RF, dtype=torch.complex64)
V_D = torch.tensor(V_D, dtype=torch.complex64)
# </editor-fold>

# <editor-fold desc="training process">
variable_name = f"WSR_List_MLMO_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}"
file_name = f'./WSR_List_MLMO_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}.mat'
globals()[variable_name] = torch.zeros(nr_of_training, External_iteration)  # record the WSR of each sample
# Iterate and optimize each sample
MLMO_run_time = 0

for item_index in range(nr_of_training):
    check_cpu_memory()

    maxi = 0  # 重置最大值
    # while maxi < 5.3:  # 如果最大 WSR 小于 6.8，则重跑外循环
        # maxi = 0
        # 初始化优化器
    optimizer_vd = MetaOptimizerVd(input_size_vd, hidden_size_vd, output_size_vd)
    adam_vd = torch.optim.Adam(optimizer_vd.parameters(), lr=optimizer_lr_vd)

    optimizer_vrf = MetaOptimizerVrf(input_size_vrf, hidden_size_vrf, output_size_vrf)
    adam_vrf = torch.optim.Adam(optimizer_vrf.parameters(), lr=optimizer_lr_vrf)

    mm_Wave_Channel = H[item_index, :, :].to(torch.complex64)
    # Vrf = V_RF[item_index, :, :].to(torch.complex64)
    # Vrf_init = Vrf
    # Vd = V_D[item_index, :, :, :].to(torch.complex64)
    # Vd = Vd.squeeze(1)
    # transmitter_precoder_init = Vd
    # transmitter_precoder = transmitter_precoder_init
    Vrf = torch.exp(1j * torch.rand(nr_of_BS_antennas, nr_of_rfs) * 2 * torch.pi)
    Vrf_init = Vrf
    Vd = init_transmitter_precoder(total_power, Vrf, mm_Wave_Channel)
    transmitter_precoder_init = Vd
    transmitter_precoder = transmitter_precoder_init

    LossAccumulated_vd = 0  # 记录预编码矩阵元学习网络的累积损失
    LossAccumulated_vrf = 0  # 记录相移矩阵元学习网络的累积损失
    start_time = time.time()

    for epoch in range(External_iteration):

        loss_vrf, sum_loss_vrf, Vrf = meta_learner_vrf(optimizer_vrf, 2,
                                                       regulated_user_weights, mm_Wave_Channel,
                                                       Vd.clone().detach(),  # 克隆预编码矩阵
                                                       Vrf_init  # 从头更新相移矩阵
                                                       )

        loss_vd, sum_loss_vd, Vd = meta_learner_vd(optimizer_vd, Internal_iteration,
                                                 regulated_user_weights, mm_Wave_Channel,
                                                 transmitter_precoder_init,  # 从头更新预编码矩阵
                                                 Vrf.clone().detach()  # 克隆相移矩阵
                                                 )

        transmitter_precoder = Vd
        normV = torch.norm(Vrf @ transmitter_precoder)  # 计算归一化前的预编码矩阵范数
        WW = math.sqrt(total_power) / normV
        transmitter_precoder = transmitter_precoder * WW

        loss_total = -compute_weighted_sum_rate(mm_Wave_Channel, transmitter_precoder, Vrf, regulated_user_weights)
        LossAccumulated_vd += loss_total
        LossAccumulated_vrf += loss_total

        WSR = compute_weighted_sum_rate(mm_Wave_Channel, transmitter_precoder, Vrf.detach(), regulated_user_weights)
        globals()[variable_name][item_index, epoch] = WSR  # 记录每个样本的 WSR
        if WSR > maxi:  # 只有当 WSR 大于当前最大值时才更新 maxi
            maxi = WSR.item()  # 记录每个样本的最大 WSR

        if epoch == External_iteration - 1:
            print('max', maxi, 'item', item_index)

        if (epoch + 1) % Update_steps == 0:
            adam_vd.zero_grad()
            adam_vrf.zero_grad()
            Average_loss_vd = LossAccumulated_vd / Update_steps
            Average_loss_vrf = LossAccumulated_vrf / Update_steps
            Average_loss_vd.backward(retain_graph=True)
            Average_loss_vrf.backward(retain_graph=True)
            adam_vd.step()
            adam_vrf.step()
            # if (epoch + 1) % 10 == 0:
            #     adam_vrf.step()

            LossAccumulated_vd = 0  # 重置累积损失
            LossAccumulated_vrf = 0  # 重置累积损失
    MLMO_run_time += time.time() - start_time
        # print("时间：", MLMO_run_time)
    print("时间：", MLMO_run_time)
    print(f"item {item_index} 完成，最大值 {maxi}")
    WSR_matrix = globals()[variable_name]
    sio.savemat(file_name,
                {variable_name: WSR_matrix.detach().numpy()})
    check_cpu_memory()

    #  save the WSR of each sample
average_MLMO_run_time = MLMO_run_time / nr_of_training
# data_to_save = {
#     variable_name: globals()[variable_name].detach().numpy(),
#     'average_MLMO_run_time': average_MLMO_run_time
# }
#
# # 保存.mat文件
# sio.savemat(file_name, data_to_save)
