import torch
import torch.multiprocessing as mp

from GlobalModel import Global_Model
from PartialModel import Partial_Model
from warehouse.funcs import *
from GPUContainer import GPU_Container
from Config import Config


def initialize_models(num_gpu, num_local):
    # initialize global model on CPU
    global_net = Net()
    global_model = Global_Model(state_dict = global_net.state_dict, capacity = num_of_gpus)
    
    # NOTE: Once partial global on i-th device processed len(coordinator[i]) local clients,
    #       it can call global_model.Incre_FedAvg(partial_global's state_dict
    return global_model


def main():
    config = Config().parse_args()

    # initialize global model
    global_model = initialize_models(config.num_gpu, config.num_local_models_per_gpu)
    coordinator = clients_coordinator(clients_list = list(range(int(config.num_users))), 
                    num_of_gpus = config.num_gpu)   

    GPU_Containers = []
    for gpu_idx, users in coordinator.items():
        GPU_Containers.append(GPUContainer(users = users, global_model=global_model, \
                                           gpu_parallel = config.num_local_models_per_gpu, 
                                           device = torch.device('cuda:'+str(gpu_idx))))

    for i in range(config.num_steps):
        print("Federated Step: ", i)
        pool = mp.Pool()
        
        assert len(GPU_Containers) == config.num_gpu
        partial_res_list = []
        for gpu_launcher in GPU_Containers:
            partial_res_list.append(gpu_launcher.launch_gpu(pool).get())

        for p in partial_res_list:
            global_model.Incre_FedAvg(p) 
            
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()



