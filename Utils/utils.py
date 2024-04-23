import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from torch_geometric.utils import to_networkx, degree, to_scipy_sparse_matrix
from scipy import sparse as sp


def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs


def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree


def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs


def split_data_ad(DS, idx, graphs, trained_label, percent, num_client):
    y = torch.cat([graph.y for graph in graphs])
    test_idx = []
    normal_class = trained_label
    abnormal_class = list(set(y.tolist()).difference(set([normal_class])))
    abnormal_num = []
    normal_class_idx = np.where(np.array(y.tolist()) == normal_class)
    normal_num = len(normal_class_idx[0])
    for ab in abnormal_class:
        abnormal_num.append(len(np.where(np.array(y.tolist()) == ab)[0]))
    train_sample_num = math.ceil(normal_num * percent)
    train_idx = random.sample(list(normal_class_idx[0]), train_sample_num)

    retain_train_idx = list(set(normal_class_idx[0]).difference(set(train_idx)))

    test_sample_num = min(min(abnormal_num), len(retain_train_idx))
    for ab in abnormal_class:
        temp_test_idx = np.where(np.array(y.tolist()) == ab)
        test_idx.extend(random.sample(list(temp_test_idx[0]), test_sample_num))
    test_idx.extend(retain_train_idx)
    np.savetxt(
        './data/TUDataset/' + DS + '/test_idx_' + str(idx) + '_' + str(trained_label) + '_' + str(num_client) + '.txt',
        test_idx, fmt='%d')
    np.savetxt(
        './data/TUDataset/' + DS + '/train_idx_' + str(idx) + '_' + str(trained_label) + '_' + str(num_client) + '.txt',
        train_idx, fmt='%d')
    return np.array(train_idx).astype(dtype=int).tolist(), np.array(test_idx).astype(dtype=int).tolist()


def split_data_ad_multi(DS, graphs, trained_label, percent):
    y = torch.cat([graph.y for graph in graphs])
    test_idx = []
    normal_class = trained_label
    abnormal_class = list(set(y.tolist()).difference(set([normal_class])))
    abnormal_num = []
    normal_class_idx = np.where(np.array(y.tolist()) == normal_class)
    normal_num = len(normal_class_idx[0])
    for ab in abnormal_class:
        abnormal_num.append(len(np.where(np.array(y.tolist()) == ab)[0]))
    train_sample_num = math.ceil(normal_num * percent)
    train_idx = random.sample(list(normal_class_idx[0]), train_sample_num)

    retain_train_idx = list(set(normal_class_idx[0]).difference(set(train_idx)))

    test_sample_num = min(min(abnormal_num), len(retain_train_idx))
    for ab in abnormal_class:
        temp_test_idx = np.where(np.array(y.tolist()) == ab)
        test_idx.extend(random.sample(list(temp_test_idx[0]), test_sample_num))
    test_idx.extend(retain_train_idx)
    np.savetxt(
        './data/TUDataset/' + DS + '/test_idx_' + str(normal_class) + '.txt',
        test_idx, fmt='%d')
    np.savetxt(
        './data/TUDataset/' + DS + '/train_idx_' + str(normal_class) + '.txt',
        train_idx, fmt='%d')
    return np.array(train_idx).astype(dtype=int).tolist(), np.array(test_idx).astype(dtype=int).tolist()


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes / numGraphs, numEdges / numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df


def obtain_avg_result(client_AUC, client_F1, DS):
    global hist_auc
    global hist_f1
    avg_AUC = np.around([np.mean(np.array(client_AUC))], decimals=4)
    avg_F1 = np.around([np.mean(np.array(client_F1))], decimals=4)
    if hist_auc < avg_AUC:
        hist_auc = avg_AUC
        hist_f1 = avg_F1
        AUCList = client_AUC
        F1_List = client_F1
        with open('./result_client/' + 'LG_FGAD' + '_' + DS + '_result.txt', 'a') as f:
            f.write('Avg AUC:' + str(avg_AUC) + '\n')
            f.write('Clent AUC:' + str(AUCList) + '\n')
            f.write('Clent F1:' + str(F1_List) + '\n')
    return hist_auc, hist_f1


def init_metric():
    global hist_auc
    global hist_f1
    hist_auc = -1.
    hist_f1 = -1.


def init_structure_encoding(args, gs, type_init):
    if type_init == 'rw':
        for g in gs:
            # Geometric diffusion features with Random Walk
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv = sp.diags(D)
            RW = A * Dinv
            M = RW

            SE_rw = [torch.from_numpy(M.diagonal()).float()]
            M_power = M
            for _ in range(args.n_rw - 1):
                M_power = M_power * M
                SE_rw.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw = torch.stack(SE_rw, dim=-1)

            g['stc_enc'] = SE_rw

    elif type_init == 'dg':
        for g in gs:
            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i, int(g_dg[i] - 1)] = 1

            g['stc_enc'] = SE_dg

    elif type_init == 'rw_dg':
        for g in gs:
            # SE_rw
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv = sp.diags(D)
            RW = A * Dinv
            M = RW

            SE = [torch.from_numpy(M.diagonal()).float()]
            M_power = M
            for _ in range(args.n_rw - 1):
                M_power = M_power * M
                SE.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw = torch.stack(SE, dim=-1)

            # PE_degree
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i, int(g_dg[i] - 1)] = 1

            g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    return gs
