def get_dataloader():
    from torchvision import datasets, transforms
    trans_mnist = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST(
        './data/MNIST/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(
        './data/MNIST/', train=False, download=True, transform=trans_mnist)
    return dataset_train, dataset_test


def lunch_training_on_different_gpu(model):
    num_users = 20
    num_of_gpus = 2
    num_of_models_per_gpu = 2
    model = model
    # In order to make the multiprocess work with cuda, need to add this line.
    mp.set_start_method('spawn', force=True)
    results = []   # results container
    for group in np.arange(0, num_users, num_of_gpus*num_of_models_per_gpu):
        # another multiprocessing version.
        # processes = []
        # for group_index, user_idx in enumerate(range(group, group + num_of_gpus)):
        # 	p = mp.Process(target = train_local, \
        # 	args=(user_idx, group_index, client_VAE_containers[group_index], dataset_train, dict_users, w_glob))
        # 	p.start()
        # 	processes.append(p)
        # for p in processes:
        # 	p.join()
        #results = []
        pool = mp.Pool()  # Create a pool for processes
        for group_index, user_idx in enumerate(range(group, group + num_of_gpus*num_of_models_per_gpu)):
            result = pool.apply_async(train_local,
                                      args=(user_idx, group_index//num_of_models_per_gpu, Global_VAE, dataloader))  # here we call func::train_local
            results.append(result)
            #count = count + num_of_gpus*num_of_models_per_gpu

        pool.close()  # can't add new process to this pool
        pool.join()  # wait for everyone to end


def train_local(user_index, gpu_index, local_model_container, global_model, dataloader):
    model = local_model_container(init_from(true_global))
    for x, y in dataloader:
        x.to(gpu_index)
        y.to(gpu_index)
        loss = cross_entropy(y, model(x))

        blablabla
    return model


def FedAvg(w):
        # w is a list of state_dict
    w_avg = w[0]
    for k in w_avg.keys():  # iterate every weight element
        for i in range(1, len(w)):  # sum through every user's weight
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def Incre_FedAvg(w_in, w_avg, counter):
    # w_in is state_dict of a new trained local model
    # w_avg is state_dict of partial global
    for k in w_avg.keys():  # iterate every weight element
        w_avg[k] += torch.div(w_in[k], counter)
        w_avg[k] = torch.div(w_avg[k], 1/counter+1)
    return w_avg
