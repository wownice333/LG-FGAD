import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from Utils.compute_metric import compute_pre_recall_f1
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation
from model.models import RKDLoss


class Client_GC():
    def __init__(self, tea_GIN, stu_GIN, sd_model, tea_head, stu_head, client_id, client_name, train_size, dataLoader,
                 optimizer_G, optimizer_D, args):
        self.tea_encoder = tea_GIN.to(args.device)
        self.stu_encoder = stu_GIN.to(args.device)
        self.tea_discriminator = tea_head.to(args.device)
        self.stu_discriminator = stu_head.to(args.device)
        self.sd_model = sd_model.to(args.device)

        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.args = args
        self.clip_value = args.clip_value

        self.W = {key: value for key, value in self.stu_encoder.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.stu_encoder.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.stu_encoder.named_parameters()}

        self.W_stu = {key: value for key, value in self.stu_discriminator.named_parameters()}
        self.dW_stu = {key: torch.zeros_like(value) for key, value in self.stu_discriminator.named_parameters()}
        self.W_old_stu = {key: value.data.clone() for key, value in self.stu_discriminator.named_parameters()}

        self.gconvNames = None

        self.gconvNames_stu = None
        self.gconvNames_sharedGIN = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, server):
        # self.gconvNames = server.W_enc.keys()
        for k in server.W_dis:
            self.W_stu[k].data = server.W_dis[k].data.clone()
        for k in server.W_enc:
            self.W[k].data = server.W_enc[k].data.clone()

    def cache_weights(self):
        for name in self.W_stu.keys():
            self.W_old_stu[name].data = self.W_stu[name].data.clone()
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W_stu, source=self.W_old_stu, keys=self.gconvNames)
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def local_train(self, local_epoch, beta, gamma):

        train_stats = train_gc(self.tea_encoder, self.stu_encoder, self.tea_discriminator, self.stu_discriminator,
                               self.sd_model,
                               self.dataLoader,
                               self.optimizer_G, self.optimizer_D, self.args.temp, local_epoch,
                               self.args.device, beta, gamma, self.clip_value, self.args.loss_mode)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W_stu)).item()

    def evaluate(self):
        return eval_gc(self.stu_encoder, self.stu_discriminator, self.dataLoader['test'],
                       self.args.device)

    def save(self, idx, avg_AUC):
        torch.save({'tea_encoder': self.tea_encoder.state_dict(), 'stu_encoder': self.stu_encoder.state_dict(),
                    'stu_discriminator': self.stu_discriminator.state_dict(),
                    'tea_discriminator': self.tea_discriminator.state_dict(),
                    'self_boosted_generator': self.sd_model.state_dict()},
                   './weight/' + self.args.data_group + '/Client_' + str(idx) + '_' + str(avg_AUC) + '.pth')


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])


def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm


def train_gc(tea_encoder, stu_encoder, tea_discriminator, stu_discriminator, sd_model, dataloaders, optimizer_G,
             optimizer_D, temp,
             local_epoch, device, beta, gamma, clip_value, loss_mode=0):
    losses_train = []
    rkd_fn = RKDLoss()
    train_loader, test_loader = dataloaders['train'], dataloaders['test']
    for epoch in range(local_epoch):
        tea_encoder.train()
        stu_encoder.train()
        stu_discriminator.train()
        tea_discriminator.train()
        sd_model.train()
        total_loss = 0.
        ngraphs = 0
        for _, batch in enumerate(train_loader):
            data = batch.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch

            if data.x is None or len(data.x.shape) == 1 or data.x.shape[1] == 0:
                adj = torch.zeros(data.batch.shape[0], data.batch.shape[0])
            else:
                adj = torch.zeros(data.x.shape[0], data.x.shape[0])
            # Restore adj from edge_index
            adj[data.edge_index] = 1
            adj = adj.to(device)

            if epoch % 10 == 0:
                # Training Generator
                optimizer_G.zero_grad()
                fake_X, fake_A = sd_model(x, edge_index, batch)
                fake_z, _ = tea_encoder(fake_X, fake_A, data.batch)
                loss_G = -torch.mean(
                    tea_discriminator(fake_z)) + 0.0001 * sd_model.latent_loss(sd_model.mean,
                                                                               sd_model.logstd)
                loss_G.backward()
                optimizer_G.step()
            else:
                loss_G = 0

            # Training Discriminator
            optimizer_D.zero_grad()
            with torch.no_grad():
                reconstructed_x, A_tilde = sd_model(x, edge_index, batch)
            tea_z, node_tea_z = tea_encoder(data.x, adj, data.batch)
            fake_tea_z, fake_node_tea_z = tea_encoder(reconstructed_x, A_tilde, data.batch)
            pred_tea = tea_discriminator(tea_z)
            loss_D_tea = -torch.mean(pred_tea) + torch.mean(
                tea_discriminator(fake_tea_z))

            stu_z, _ = stu_encoder(data.x, data.edge_index, data.batch)
            pred_stu = stu_discriminator(stu_z)

            loss_D_stu = -torch.mean(pred_stu)
            measure = 'JSD'
            loss_D = loss_D_tea + loss_D_stu + gamma * (com_distillation_loss(pred_tea, pred_stu, temp,
                                                                              loss_mode) + rkd_fn(stu_z,
                                                                                                  tea_z)) + beta * (
                                 local_global_loss_(
                                     node_tea_z, tea_z, batch, measure) + global_loss_(fake_tea_z, tea_z,
                                                                                       measure) + local_global_loss_(
                             fake_node_tea_z, fake_tea_z, batch, measure))
            loss_D.backward()
            optimizer_D.step()
            # Clip weights of discriminator
            for p in tea_discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)
            for p in stu_discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)
            total_loss += loss_D + loss_G
            ngraphs += data.num_graphs

        total_loss /= ngraphs

        losses_train.append(total_loss)
    return {'trainingLosses': losses_train}


def eval_gc(stu_encoder, stu_discriminator, test_loader, device):
    stu_encoder.eval()
    stu_discriminator.eval()
    ngraphs = 0
    label_score = []
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            z, _ = stu_encoder(batch.x, batch.edge_index, batch.batch)
            pred = stu_discriminator(z)  # .squeeze()
            sigmoid_scores = torch.sigmoid(pred)

            label = batch.y.float()
        label_score += list(zip(label.cpu().data.numpy().tolist(),
                                pred.cpu().data.numpy().tolist(),
                                sigmoid_scores.cpu().data.numpy().tolist(),
                                z.cpu().data.numpy().tolist()))
        ngraphs += batch.num_graphs

    labels, scores, sigmoid_scores, middle = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores).squeeze()
    sigmoid_scores = np.array(sigmoid_scores).squeeze()
    middle = np.array(middle)

    labels = np.where(labels == 0, 1, 0)
    test_f1, test_recall = compute_pre_recall_f1(labels, scores)

    test_auc = roc_auc_score(labels, scores)
    return test_auc, test_f1


# KL-divergence Loss for knowledge distillation
def com_distillation_loss(t, s, temp, loss_mode):
    s_dist = F.log_softmax(s / temp)
    t_dist = F.softmax(t / temp)
    if loss_mode == 0:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist)
    elif loss_mode == 1:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())
    return kd_loss


def local_global_loss_(l_enc, g_enc, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes

    return - E_pos


def global_loss_(g_enc_1, g_enc_2, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc_1.shape[0]

    res = torch.mm(g_enc_1, g_enc_2.t())
    E_neg = get_negative_expectation(res, measure, average=False).sum()
    E_neg = E_neg / ((num_graphs - 1) * (num_graphs - 1))

    return E_neg
