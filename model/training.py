from Utils.utils import obtain_avg_result, init_metric


def run_lgfgad(clients, server, COMMUNICATION_ROUNDS, local_epoch, beta, gamma, DS, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    init_metric()
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        print('conmmunication rounds:', c_round)
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train(local_epoch, beta, gamma)
        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)
        client_AUC = []
        client_F1 = []
        for client in clients:
            test_auc, test_f1 = client.evaluate()
            client_AUC.append(test_auc)
            client_F1.append(test_f1)
        avg_AUC, avg_F1 = obtain_avg_result(client_AUC, client_F1, DS)
    return avg_AUC, avg_F1


