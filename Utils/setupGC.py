import os
import random
from itertools import chain
from random import choices

import pandas as pd
import torch
from numpy import loadtxt
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree

from Utils.utils import get_maxDegree, get_stats, get_numGraphLabels, split_data_ad, split_data_ad_multi, \
    init_structure_encoding
from model.client import Client_GC
from model.models import *
from model.server import Server


def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum / num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i * minSize:(i + 1) * minSize])
        for g in graphs[num_client * minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks


def prepareData_oneDS(datapath, data, num_client, batchSize, percentage, convert_x=False, seed=None,
                      overlap=False):
    print('Client Num', num_client)
    normal_class = 0
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    graphs = [x for x in tudataset]
    print("  **", data, len(graphs))

    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        if not os.path.exists(
                './data/TUDataset/' + data + '/test_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt') or not os.path.exists(
            './data/TUDataset/' + data + '/train_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                num_client) + '.txt'):
            print('Split Data')
            train_idx, test_idx = split_data_ad(data, idx, ds_tvt, normal_class, percentage, num_client)
        else:
            train_idx = np.array(
                (loadtxt('./data/TUDataset/' + data + '/train_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt'))).astype(
                dtype=int).tolist()
            test_idx = np.array(
                (loadtxt('./data/TUDataset/' + data + '/test_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt'))).astype(
                dtype=int).tolist()
        print(train_idx)
        ds_train = [ds_tvt[i] for i in train_idx]
        ds_test = [ds_tvt[i] for i in test_idx]
        ds_val = ds_test
        dataloader_train = DataLoader(ds_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(ds_val, batch_size=batchSize, shuffle=False)
        dataloader_test = DataLoader(ds_test, batch_size=batchSize, shuffle=False)
        num_graph_labels = get_numGraphLabels(ds_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(ds_train))
        df = get_stats(df, ds, ds_train, graphs_val=ds_val, graphs_test=ds_test)

    return splitedData, df


def prepareData_multiDS(datapath, normal_class, percentage, group='small', batchSize=32, convert_x=False, seed=None):
    assert group in ['molecules', 'molecules_tiny', 'small', 'mix', "mix_tiny", "biochem", "biochem_tiny", "socialnet"]

    if group == 'molecules' or group == 'molecules_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    if group == 'small':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]  # bioinformatics
    if group == 'mix' or group == 'mix_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS",  # bioinformatics
                    "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]  # social networks
    if group == 'biochem' or group == 'biochem_tiny':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",  # small molecules
                    "ENZYMES", "DD", "PROTEINS"]  # bioinformatics
    if group == 'socialnet':
        datasets = ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "COLLAB":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
        elif data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)
            if convert_x:
                maxdegree = get_maxDegree(tudataset)
                tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
        print('Normal class:', normal_class)
        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))
        if not os.path.exists(
                './data/TUDataset/' + data + '/test_idx_' + str(
                    normal_class) + '.txt') or not os.path.exists(
            './data/TUDataset/' + data + '/train_idx_' + '_' + str(normal_class) + '.txt'):
            print('Split Data')
            train_idx, test_idx = split_data_ad_multi(data, graphs, normal_class, percentage)

        else:
            train_idx = np.array(
                (loadtxt(
                    './data/TUDataset/' + data + '/train_idx_' + str(normal_class) + '.txt'))).astype(
                dtype=int).tolist()
            test_idx = np.array(
                (loadtxt(
                    './data/TUDataset/' + data + '/test_idx_' + str(normal_class) + '.txt'))).astype(
                dtype=int).tolist()
        print(train_idx)

        graphs_train = [graphs[i] for i in train_idx]
        graphs_test = [graphs[i] for i in test_idx]
        graphs_val = graphs_test
        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=False)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=False)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                             num_node_features, num_graph_labels, len(graphs_train))

        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
    return splitedData, df


def prepareData_oneDS_Fedstar(args, datapath, data, normal_class, percentage, num_client, batchSize, convert_x=False,
                              seed=None,
                              overlap=False):
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    print('normal class:', normal_class)
    graphs = [x for x in tudataset]
    print("  **", data, len(graphs))

    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        if not os.path.exists(
                './data/TUDataset/' + data + '/test_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt') or not os.path.exists(
            './data/TUDataset/' + data + '/train_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                num_client) + '.txt'):
            print('Split Data')
            train_idx, test_idx = split_data_ad(data, idx, ds_tvt, normal_class, percentage, num_client)
        else:
            train_idx = np.array(
                (loadtxt('./data/TUDataset/' + data + '/train_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt'))).astype(
                dtype=int).tolist()
            test_idx = np.array(
                (loadtxt('./data/TUDataset/' + data + '/test_idx_' + str(idx) + '_' + str(normal_class) + '_' + str(
                    num_client) + '.txt'))).astype(
                dtype=int).tolist()
        print(train_idx)
        graphs_train = [ds_tvt[i] for i in train_idx]
        graphs_test = [ds_tvt[i] for i in test_idx]
        graphs_val = graphs_test

        graphs_train = init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)
        graphs_val = init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)
        graphs_test = init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=False)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=False)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=False)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)
        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(graphs_train))
        df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        if num_node_features == 0:
            num_node_features = 1
        cmodel_gc = Graph_Representation_Learning(args.hidden, args.nlayer, num_node_features, args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        clients.append(Client_GC_Baseline(cmodel_gc, idx, ds, train_size, dataloaders, optimizer, args))
    smodel = server_graph_representation_learning(args.hidden, args.nlayer, num_node_features, args.device)
    server = Server_Baseline(smodel, args.device)
    return clients, server, idx_clients


def setup_devices(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        if num_node_features == 0:
            num_node_features = 1
        stu_GIN = Encoder(num_node_features, args.hidden, args.stu_nlayer, args.device)
        tea_GIN = backbone_GIN(num_node_features, args.hidden, args.tea_nlayer, args.device)
        cgen_gc = Co_VGAE(num_node_features, args.hidden, args.gen_nlayer, args.device)
        cteacher = teacher_head(args.hidden, args.device)
        cstudent = student_head(args.hidden * args.stu_nlayer, args.device)
        # Combine the parameters of both models
        combined_parameters = chain(stu_GIN.parameters(), tea_GIN.parameters(), cstudent.parameters(),
                                    cteacher.parameters())
        generator_parameters = chain(cgen_gc.parameters())

        optimizer_G = torch.optim.Rprop(filter(lambda p: p.requires_grad, generator_parameters),
                                        lr=args.lr)
        optimizer_D = torch.optim.Rprop(filter(lambda p: p.requires_grad, combined_parameters), lr=args.lr)
        clients.append(
            Client_GC(tea_GIN, stu_GIN, cgen_gc, cteacher, cstudent, idx, ds, train_size, dataloaders,
                      optimizer_G, optimizer_D, args))

    smodel_dis = student_head(args.hidden * args.stu_nlayer, args.device)
    smodel_enc = Encoder(num_node_features, args.hidden, args.stu_nlayer, args.device)
    server = Server(smodel_enc, smodel_dis, args.device)
    return clients, server, idx_clients


def setup_devices_multi(splitedData, args):
    idx_clients = {}
    clients = []
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        if num_node_features == 0:
            num_node_features = 1
        stu_GIN = Encoder_multi(num_node_features, args.hidden, args.stu_nlayer, args.device)
        tea_GIN = backbone_GIN(num_node_features, args.hidden, args.tea_nlayer, args.device)
        cgen_gc = Co_VGAE(num_node_features, args.hidden, args.gen_nlayer, args.device)
        cteacher = teacher_head(args.hidden, args.device)
        cstudent = student_head(args.hidden * args.stu_nlayer, args.device)
        # Combine the parameters of both models
        combined_parameters = chain(stu_GIN.parameters(), tea_GIN.parameters(), cstudent.parameters(),
                                    cteacher.parameters())
        generator_parameters = chain(cgen_gc.parameters())

        optimizer_G = torch.optim.Rprop(filter(lambda p: p.requires_grad, generator_parameters),
                                        lr=args.lr)
        optimizer_D = torch.optim.Rprop(filter(lambda p: p.requires_grad, combined_parameters), lr=args.lr)

        clients.append(
            Client_GC(tea_GIN, stu_GIN, cgen_gc, cteacher, cstudent, idx, ds, train_size, dataloaders,
                      optimizer_G, optimizer_D, args))

    smodel_dis = student_head(args.hidden * args.stu_nlayer, args.device)
    smodel_enc = Server_encoder_multi(num_node_features, args.hidden, args.stu_nlayer, args.device)
    server = Server(smodel_enc, smodel_dis, args.device)
    return clients, server, idx_clients
