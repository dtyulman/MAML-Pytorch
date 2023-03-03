import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        loss_q = 0
        acc_q = [0 for _ in range(self.update_step + 1)] #k-th item is acc on step k=0~self.update_step
        for i in range(task_num):
            #K steps of optimizing per-task parameters (theta_prime_i) for i-th task
            loss_q_i, acc_q_i = self.inner_loop(self.net, x_spt[i], y_spt[i], x_qry[i], y_qry[i], self.update_step)

            #accumulate loss/acc over tasks
            loss_q += loss_q_i
            acc_q = [a + ai for a,ai in zip(acc_q, acc_q_i)]

        #normalize loss, acc
        loss_q = loss_q / task_num
        acc_q = np.array(acc_q) / (querysz * task_num)

        #one step of optimizing meta parameters (theta)
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        return acc_q


    def inner_loop(self, net, x_spt, y_spt, x_qry, y_qry, n_steps):
        acc_q = []
        with torch.no_grad(): #acc before update
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            acc_q.append( torch.eq(logits_q.argmax(dim=1), y_qry).sum().item() )

        fast_weights = net.parameters()
        for k in range(n_steps):
            #loss on support set
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)

            #compute grad on theta_pi and update theta_pi = theta_pi - train_lr * grad
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            #acc on query set at step k
            with torch.no_grad():
                logits_q = net(x_qry, fast_weights, bn_training=True)
                acc_q.append( torch.eq(logits_q.argmax(dim=1), y_qry).sum().item() )

        #final loss on query set
        logits_q = net(x_qry, fast_weights, bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry)

        return loss_q, acc_q


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        _, acc_q = self.inner_loop(net, x_spt, y_spt, x_qry, y_qry, self.update_step_test)
        del net

        acc_q = np.array(acc_q) / querysz
        return acc_q
