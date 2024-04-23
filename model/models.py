import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU
from torch_geometric.nn import GINConv, global_add_pool


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, w_d=1, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = 0

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # # RKD Angle loss
        # with torch.no_grad():
        #     td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        #     norm_td = F.normalize(td, p=2, dim=2)
        #     t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        #
        # sd = (student.unsqueeze(0) - student.unsqueeze(1))
        # norm_sd = F.normalize(sd, p=2, dim=2)
        # s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        #
        # loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d  # + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


class teacher_head(torch.nn.Module):
    def __init__(self, dim, device):
        super(teacher_head, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(dim * 4, dim * 3)
        self.fc2 = nn.Linear(dim * 3, dim * 2)
        self.fc3 = nn.Linear(dim * 2, dim)
        self.fc4 = nn.Linear(dim, 1)

    def forward(self, g_enc):
        x = F.leaky_relu(self.fc1(g_enc))
        x = F.leaky_relu(self.fc2(x))
        middle = F.leaky_relu(self.fc3(x))
        output = self.fc4(middle)
        return output


class student_head(torch.nn.Module):
    def __init__(self, dim, device):
        super(student_head, self).__init__()
        self.device = device
        # self.fc1 = nn.Linear(dim, int(dim/2))
        # self.fc2 = nn.Linear(int(dim/2), int(dim/4))
        # self.fc3 = nn.Linear(int(dim/4), 1)
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc1 = nn.Linear(dim, 16)
        # self.fc2 = nn.Linear(16, 8)
        # self.fc3 = nn.Linear(8, 1)

    def forward(self, g_enc):
        middle = F.leaky_relu(self.fc1(g_enc))
        middle = F.leaky_relu(self.fc2(middle))
        output = self.fc3(middle)
        return output


class Co_VGAE(nn.Module):
    def __init__(self, num_features, latent_dim, num_gc_layers, device, alpha=0.5, beta=1., gamma=.1):
        super(Co_VGAE, self).__init__()
        self.dataset_num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        # print(dataset_num_features)
        self.hidden_dim = latent_dim
        self.num_gc_layers = num_gc_layers
        self.embedding_dim = mi_units = self.hidden_dim * self.num_gc_layers
        self.base_gcn = Encoder(self.dataset_num_features, self.hidden_dim, self.num_gc_layers, self.device)
        self.gcn_mean = Encoder(self.embedding_dim, self.hidden_dim, self.num_gc_layers, self.device)
        self.gcn_logstddev = Encoder(self.embedding_dim, self.hidden_dim, self.num_gc_layers, self.device)
        self.decoder = Sequential(Linear(self.embedding_dim, math.ceil(self.dataset_num_features / 2)), LeakyReLU(),
                                  Linear(math.ceil(self.dataset_num_features / 2), self.dataset_num_features))

    def encode(self, x, edge_index, batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones(batch.shape[0]).to(self.device)
        # print(x.shape)
        _, hidden = self.base_gcn(x, edge_index, batch)
        # print(hidden.shape)
        _, self.mean = self.gcn_mean(hidden, edge_index, batch)
        _, self.logstd = self.gcn_logstddev(hidden, edge_index, batch)
        # hidden = self.base_gcn(x)
        # self.mean = self.gcn_mean(hidden)
        # self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(batch.shape[0], self.hidden_dim * self.num_gc_layers).to(self.device)
        # print(self.logstd.shape)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, x, edge_index, batch):
        z = self.encode(x, edge_index, batch)
        reconstructed_x = self.decoder(z)
        A_pred = dot_product_decode(z)
        return reconstructed_x, A_pred

    def latent_loss(self, z_mean, z_stddev):
        kl_divergence = 0.5 * torch.sum(torch.exp(z_stddev) + torch.pow(z_mean, 2) - 1. - z_stddev)
        return kl_divergence / z_mean.size(0)

    def loss(self, x_rec, x):
        reconstruction_loss = F.mse_loss(x_rec, x)
        kl_loss_node = self.latent_loss(self.mean, self.logstd)
        total_loss = reconstruction_loss + 0.001 * kl_loss_node
        return total_loss


class GraphConv(nn.Module):
    def __init__(self):
        super(GraphConv, self).__init__()

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        return x


class backbone_GIN(torch.nn.Module):
    def __init__(self, num_features, latent_dim, num_gc_layers, device):
        super(backbone_GIN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device
        # self.nns = []
        # self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.nns = torch.nn.ModuleList()
        self.embedding_dim = latent_dim * num_gc_layers
        self.gin_conv = GraphConv()
        for i in range(self.num_gc_layers):
            bn = torch.nn.BatchNorm1d(latent_dim, eps=1e-04, affine=False, track_running_stats=True)
            if i:
                nn = Sequential(Linear(latent_dim, latent_dim), LeakyReLU(), Linear(latent_dim, latent_dim))
            else:
                nn = Sequential(Linear(num_features, latent_dim), LeakyReLU(), Linear(latent_dim, latent_dim))

            self.nns.append(nn)
            self.bns.append(bn)
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.fill_(0.0)
                torch.nn.init.xavier_uniform_(m.weight.data)
                # if m.bias is not None:
                #     m.bias.data.fill_(0.0)

    def forward(self, x, adj, batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = self.gin_conv(x, adj)
            x = self.nns[i](x)
            x = self.bns[i](x)
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        x_global = torch.cat(xpool, 1)
        x_node = torch.cat(xs, 1)
        g_enc = x_global
        l_enc = x_node
        return g_enc, l_enc


class Server_encoder_multi(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Server_encoder_multi, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device
        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
            # nn = Sequential(Linear(dim, dim, bias=False))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.bns.append(bn)


class Encoder_multi(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Encoder_multi, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device
        self.pre = torch.nn.Sequential(torch.nn.Linear(num_features, dim))
        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
            # nn = Sequential(Linear(dim, dim, bias=False))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        x = self.pre(x)
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim, bias=True), LeakyReLU(), Linear(dim, dim, bias=True))
            else:
                nn = Sequential(Linear(num_features, dim, bias=True), LeakyReLU(), Linear(dim, dim, bias=True))
            # nn = Sequential(Linear(dim, dim, bias=False))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)

            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)
