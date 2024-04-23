import argparse
import copy
import random
import warnings

import torch

from Utils import setupGC
from model.training import *

warnings.filterwarnings("ignore")


def process_lgfgad(clients, server, args):
    print("\nDone setting up LG-FGAD devices.")

    print("Running LG-FGAD ...")
    if args.eval == True:
        idx = 0
        for client in clients:
            client.stu_encoder.load_state_dict(
                torch.load("./weight/" + args.data_group + "/Client_" + str(idx) + ".pth")['stu_encoder'])
            client.stu_discriminator.load_state_dict(
                torch.load("./weight/" + args.data_group + "/Client_" + str(idx) + ".pth")['stu_discriminator'])
            idx += 1
        client_AUC = []
        client_F1 = []
        idx = 0
        for client in clients:
            test_auc, test_f1 = client.evaluate()
            client_AUC.append(test_auc)
            client_F1.append(test_f1)
            idx += 1
        AUC = np.around([np.mean(np.array(client_AUC))], decimals=4)
        F1 = np.around([np.mean(np.array(client_F1))], decimals=4)
    else:
        AUC, F1, = run_lgfgad(clients, server, args.num_rounds, args.local_epoch, args.beta, args.gamma,
                              args.data_group, samp=None)
    return AUC, F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=10,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--stu_nlayer', type=int, default=2,
                        help='Number of student GINconv layers')
    parser.add_argument('--tea_nlayer', type=int, default=4,
                        help='Number of student GINconv layers')
    parser.add_argument('--gen_nlayer', type=int, default=3,
                        help='Number of student GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='MUTAG')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=10)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.01)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=1.0, help='temperature of Knowledge Distillation')
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=5)
    parser.add_argument('--loss_mode', type=int, default=0)
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                        help='normal class index')
    parser.add_argument('--beta', type=float, default=10, metavar='N',
                        help='Weight of the mutual information loss')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='N',
                        help='Weight of the knowledge distillation loss')
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")
    parser.add_argument('--eval', help='whether load the saved model to reproduce results',
                        type=bool, default=False)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # args.device = "cpu"

    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2

    """ distributed one dataset to multiple clients """

    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")
    percentage = 0.8
    splitedData, df_stats = setupGC.prepareData_oneDS(args.datapath, args.data_group,
                                                      num_client=args.num_clients, batchSize=args.batch_size,
                                                      percentage=percentage,
                                                      convert_x=args.convert_x, seed=seed_dataSplit,
                                                      overlap=args.overlap)
    print("Done")
    repNum = args.num_repeat
    hist_AUC = []
    hist_F1 = []
    for epoch in range(repNum):
        init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
        print("\nDone setting up devices.")
        AUC, F1 = process_lgfgad(clients=copy.deepcopy(init_clients),
                                 server=copy.deepcopy(init_server), args=args)
        hist_AUC.append(AUC)
        hist_F1.append(F1)
    Mean_AUC = np.around([np.mean(np.array(hist_AUC)), np.std(np.array(hist_AUC))], decimals=4)
    Mean_F1 = np.around([np.mean(np.array(hist_F1)), np.std(np.array(hist_F1))], decimals=4)
    print('Average AUC:' + str(Mean_AUC[0] * 100) + '$\pm$' + str(Mean_AUC[1] * 100))
    print('Average F1:' + str(Mean_F1[0] * 100) + '$\pm$' + str(Mean_F1[1] * 100))
    with open('./result/' + args.data_group + '_result.txt', 'a') as f:
        f.write('Average AUC:' + str(Mean_AUC[0] * 100) + '$\pm$' + str(Mean_AUC[1] * 100) + '\n')
        f.write('Average F1:' + str(Mean_F1[0] * 100) + '$\pm$' + str(Mean_F1[1] * 100) + '\n')
