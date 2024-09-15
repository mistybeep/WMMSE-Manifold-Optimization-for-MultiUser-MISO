from util import *
import gc
from efficient_kan import KAN

class MetaOptimizerVrf(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MetaOptimizerVrf, self).__init__()

        self.layer = nn.Sequential(
            # KAN([input_size, hidden_size, output_size]),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, gradient):
        """
        Implement forward propagation of the meta learning network for theta
        :param gradient: the gradient of SE with respect to theta, with sum of user weights normalized to 1
        :return: regulated delta theta
        """
        gradient = gradient.unsqueeze(0)
        gradient = self.layer(gradient)
        gradient = gradient.squeeze(0)
        return gradient


class MetaOptimizerVd(nn.Module):
    """
    this class is used to define the meta learning network for w
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        this function is used to initialize the meta learning network for w
        :param input_size: the size of the input, which is nr_of_users*2 in this code
        :param hidden_size: the size of hidden layers, which is hidden_size_w in this code
        :param output_size: the size of the output, which is nr_of_users*2 in this code
        """
        super(MetaOptimizerVd, self).__init__()

        self.layer = nn.Sequential(
            # KAN([input_size, hidden_size, output_size]),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, gradient):
        """
        this function is used to implement the forward propagation of the meta learning network for w
        :param gradient: the gradient of SE with respect to w, with sum of user weights normalized to 1
        :return: delta w
        """
        gradient = gradient.unsqueeze(0)
        gradient = self.layer(gradient)
        gradient = gradient.squeeze(0)
        return gradient


# </editor-fold>


# <editor-fold desc="build meta-learners">

def meta_learner_vd(optimizee, Internal_iteration, user_weights, Channel, Vd,
                    Vrf, retain_graph_flag=True):
    """
    Implementation of inner iteration of meta learning for w
    :param optimizee: optimizer for w
    :param Internal_iteration: number of inner loops in each outer loop
    :param user_weights: the weight of each user
    :param channel1: channel G
    :param X: the compressed precoding matrix
    :param theta: the phase shift matrix
    :param noise_power: the noise power
    :param retain_graph_flag: whether to retain the graph
    :return: the loss, the accumulated loss, and the updated compressed precoding matrix
    """
    L_vd = 0
    Vd_internal = Vd  # initialize the compressed precoding matrix
    Vd_internal.requires_grad = True  # set the requires_grad flag to true to enable the backward propagation
    sum_loss_vd = 0  # record the accumulated loss
    for internal_index in range(Internal_iteration):
        L_vd = - compute_weighted_sum_rate(Channel, Vd_internal, Vrf, user_weights)
        sum_loss_vd = L_vd + sum_loss_vd  # accumulate the loss
        sum_loss_vd.backward(retain_graph=retain_graph_flag)  # compute the gradient
        Vd_grad = Vd_internal.grad.clone().detach()  # clone the gradient
        #  as pytorch can not process complex number, we have to split the real and imaginary parts and concatenate them
        Vd_grad1 = torch.cat((Vd_grad.real, Vd_grad.imag), dim=1)  # concatenate the real and imaginary part
        shape = Vd_grad1.shape
        Vd_grad1 = Vd_grad1.reshape(-1)
        Vd_update = optimizee(Vd_grad1)  # input the gradient and get the increment
        Vd_update = Vd_update.reshape(shape)
        Vd_update1 = Vd_update[:, 0: nr_of_users] + 1j * Vd_update[:, nr_of_users: 2 * nr_of_users]
        Vd_internal = Vd_internal + Vd_update1  # update the compressed precoding matrix
        Vd_update.retain_grad()
        Vd_internal.retain_grad()

    return L_vd, sum_loss_vd, Vd_internal


def meta_learner_vrf(optimizee, Internal_iteration, user_weights, Channel, Vd,
                     Vrf, retain_graph_flag=True):
    L_vrf = 0
    vrf_internal = Vrf
    vrf_internal.requires_grad = True
    sum_loss_vrf = 0
    for internal_index in range(Internal_iteration):
        L_vrf = - compute_weighted_sum_rate(Channel, Vd, vrf_internal, user_weights)
        sum_loss_vrf = L_vrf + sum_loss_vrf
        sum_loss_vrf.backward(retain_graph=retain_graph_flag)
        Vrf_grad = vrf_internal.grad.clone().detach()
        Vrf_grad1 = torch.cat((Vrf_grad.real, Vrf_grad.imag), dim=1)
        shape = Vrf_grad1.shape
        Vrf_grad1 = Vrf_grad1.reshape(-1)
        vrf_update = optimizee(Vrf_grad1)
        vrf_update = vrf_update.reshape(shape)
        vrf_update1 = vrf_update[:, 0: nr_of_rfs] + 1j * vrf_update[:, nr_of_rfs: 2 * nr_of_rfs]
        vrf_riemannian_grad = euclidean_orthogonal_projection(vrf_update1, vrf_internal)
        vrf_internal = retraction(vrf_internal, vrf_riemannian_grad)
        vrf_update.retain_grad()
        vrf_internal.retain_grad()
    return L_vrf, sum_loss_vrf, vrf_internal


# </editor-fold>

# <editor-fold desc="initialize the network and optimizer">
# initialize the meta learning network w parameters
input_size_vd = nr_of_rfs * nr_of_users * 2
hidden_size_vd = 128
output_size_vd = nr_of_rfs * nr_of_users * 2
batch_size_vd = nr_of_users

# initialize the meta learning network theta parameters
input_size_vrf = nr_of_BS_antennas * nr_of_rfs * 2
hidden_size_vrf = 128
output_size_vrf = nr_of_BS_antennas * nr_of_rfs * 2
batch_size_vrf = nr_of_rfs

# </editor-fold>


# 测试函数，仅供测试用途
# if __name__ == '__main__':
#     print("input_size_w: ", input_size_w, "\n",
#           "hidden_size_w: ", hidden_size_w, "\n",
#           "output_size_w: ", output_size_w, "\n",
#           "batch_size_w: ", batch_size_w, "\n",
#           "input_size_theta: ", input_size_theta, "\n",
#           "hidden_size_theta: ", hidden_size_theta, "\n",
#           "output_size_theta: ", output_size_theta, "\n",
#           "batch_size_theta: ", batch_size_theta, "\n",
#           )
