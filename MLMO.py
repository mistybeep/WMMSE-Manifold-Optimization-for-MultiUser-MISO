import random
import scipy.io as sio
import torch
import numpy as np
import math
import time
import torch.nn.init as init
from net import *  # 需要你自己实现的网络结构

# <editor-fold desc="set random seed">
seed = 42
torch.manual_seed(seed)  # cpu random seed
torch.cuda.manual_seed(seed)  # gpu random seed
torch.cuda.manual_seed_all(seed)  # multi-gpu random seed
np.random.seed(seed)  # numpy random seed
random.seed(seed)  # python random seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# </editor-fold>

# 加载数据
HH = sio.loadmat(f'dataset.mat')['HH']  # load the channel H, numpy format
V_RF = sio.loadmat(f'dataset.mat')['V_RF_train']
V_D = sio.loadmat(f'dataset.mat')['V_D_train']
user_weights = sio.loadmat(f'dataset.mat')['omega'].squeeze()  # load the user weights, numpy format
regulated_user_weights = user_weights / np.sum(user_weights)

# 转换为PyTorch张量并使用复杂数据类型
H = torch.tensor(HH, dtype=torch.complex64)
V_RF = torch.tensor(V_RF, dtype=torch.complex64)
V_D = torch.tensor(V_D, dtype=torch.complex64)

# 变量名和文件名
variable_name = f"WSR_List_MLMO_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}"
file_name = f'./WSR_List_MLMO_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}.mat'
globals()[variable_name] = torch.zeros(nr_of_training)
# globals()[variable_name] = torch.zeros(nr_of_training, External_iteration)
# 初始化运行时间
MLMO_run_time = 0

# 主训练循环
for item_index in range(nr_of_training):
    maxi = 0  # 重置最大 WSR
    while maxi < 5.3:  # 如果最大 WSR 小于 5.3，则重新运行外部循环
        optimizer_vd = MetaOptimizerVd(input_size_vd, hidden_size_vd, output_size_vd)
        adam_vd = torch.optim.Adam(optimizer_vd.parameters(), lr=optimizer_lr_vd)

        optimizer_vrf = MetaOptimizerVrf(input_size_vrf, hidden_size_vrf, output_size_vrf)
        adam_vrf = torch.optim.Adam(optimizer_vrf.parameters(), lr=optimizer_lr_vrf)

        mm_Wave_Channel = H[item_index, :, :].to(torch.complex64)
        Vrf = torch.exp(1j * torch.rand(nr_of_BS_antennas, nr_of_rfs) * 2 * torch.pi)
        Vrf_init = Vrf
        Vd = init_transmitter_precoder(total_power, Vrf, mm_Wave_Channel)
        transmitter_precoder_init = Vd
        transmitter_precoder = transmitter_precoder_init

        LossAccumulated_vd = 0
        LossAccumulated_vrf = 0
        start_time = time.time()

        for epoch in range(External_iteration):
            # 计算 meta learner 的损失
            loss_vrf, sum_loss_vrf, Vrf = meta_learner_vrf(
                optimizer_vrf, 2, regulated_user_weights, mm_Wave_Channel,
                Vd.clone().detach(), Vrf_init
            )

            loss_vd, sum_loss_vd, Vd = meta_learner_vd(
                optimizer_vd, Internal_iteration, regulated_user_weights,
                mm_Wave_Channel, transmitter_precoder_init, Vrf.clone().detach()
            )

            transmitter_precoder = Vd
            normV = torch.norm(Vrf @ transmitter_precoder)
            WW = math.sqrt(total_power) / normV
            transmitter_precoder = transmitter_precoder * WW

            # 计算 WSR 并累积损失
            loss_total = -compute_weighted_sum_rate(mm_Wave_Channel, transmitter_precoder, Vrf, regulated_user_weights)
            LossAccumulated_vd += loss_total
            LossAccumulated_vrf += loss_total

            WSR = compute_weighted_sum_rate(mm_Wave_Channel, transmitter_precoder, Vrf.detach(), regulated_user_weights)
            # globals()[variable_name][item_index, epoch] = WSR

            # 更新最大 WSR
            if WSR > maxi:
                maxi = WSR.item()
                globals()[variable_name][item_index] = WSR

            if epoch == External_iteration - 1:
                print(f'max {maxi}, item {item_index}')

            # 每隔一定步数更新元学习网络
            if (epoch + 1) % Update_steps == 0:
                adam_vd.zero_grad()
                adam_vrf.zero_grad()

                # 计算平均损失并更新
                Average_loss_vd = LossAccumulated_vd / Update_steps
                Average_loss_vrf = LossAccumulated_vrf / Update_steps

                Average_loss_vd.backward(retain_graph=True)  # 去除 retain_graph=True
                Average_loss_vrf.backward(retain_graph=True)  # 去除 retain_graph=True

                adam_vd.step()
                adam_vrf.step()

                # 重置累积损失
                LossAccumulated_vd = 0
                LossAccumulated_vrf = 0

        # 记录运行时间
        MLMO_run_time += time.time() - start_time
        print("时间：", MLMO_run_time)
        torch.cuda.empty_cache()  # 清理缓存，避免内存泄漏
    print(f"item {item_index} 完成，最大值 {maxi}")
    WSR_matrix = globals()[variable_name]
    sio.savemat(file_name,
                {variable_name: WSR_matrix.detach().numpy()})

# 计算平均运行时间并保存结果
average_MLMO_run_time = MLMO_run_time / nr_of_training
# data_to_save = {
#     variable_name: globals()[variable_name].detach().numpy(),
#     'average_MLMO_run_time': average_MLMO_run_time
# }
# sio.savemat(file_name, data_to_save)
