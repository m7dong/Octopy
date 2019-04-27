from GlobalModel import Global_Model
from PartialModel import Partial_Model
from warehouse.funcs import *
from GPUContainer import GPU_Container

import torch

def initialize_models(num_gpu, num_local):
    # initialize global model on CPU
    global_net = Net()
    global_model = Global_Model(state_dict = global_net.state_dict, capacity = num_of_gpus)
    
    # NOTE: Once partial global on i-th device processed len(coordinator[i]) local clients,
    #       it can call global_model.Incre_FedAvg(partial_global's state_dict)
    
    # initialize partial models on GPU
    init_dict = torch.zeros(5, 3, dtype=torch.long)
    partial_model = Partial_Model(state_dict = init_dict, capacity = num_of_local, global_model=global_model)

    return global_model, partial_model


def main():
    # load data
    dataset_train, dataset_test = get_dataloader()

    # initialize models
    num_gpu, num_local = 8, 3
    global_model, partial_model = initialize_models(num_gpu, num_local)
    coordinator = clients_coordinator(clients_list = list(range(int(20))), num_of_gpus = num_of_gpus)   

    GPU_Containers = []
    for gpu_idx, users in coordinator.items():
        GPU_Containers.append(GPUContainer(partial_model, users))


    total_rounds = 10
    for t in range(total_rounds):
        # pull global to true global on each GPU
        if t > 0:
            while global_model.incre_counter != 0:
                continue
            partial_model.pull_global()

        # step training
        launch_training_on_different_gpu()
    
        # calculate avg
        for k in partial_model.state_dict.keys():
            partial_model.state_dict[k] = torch.div(partial_model.state_dict[k], partial_model.capacity)

        # update global model on CPU
        global_model.Incre_FedAvg(partial_global.state_dict)




if __name__ == '__main__':
    main()



