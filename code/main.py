import torch
import torch.multiprocessing as mp

from GlobalModel import Global_Model
from PartialModel import Partial_Model
from warehouse.funcs import *
from GPUContainer import GPU_Container
from Config import Config
from Lenet import Net
import time

def initialize_models(num_gpus, num_local, config):
    # initialize global model on CPU
    global_net = Net()
    global_model = Global_Model(state_dict = global_net.state_dict(), capacity = config.num_users)
    
    # NOTE: Once partial global on i-th device processed len(coordinator[i]) local clients,
    #       it can call global_model.Incre_FedAvg(partial_global's state_dict
    return global_model

def launch_process_update_partial(local_model_queue, global_model, done):
    # partial_model = Partial_Model(device=device, capacity=capacity, global_model=global_model)
    while True:   # scan the queue
        if not local_model_queue.empty():  
            local_model = local_model_queue.get(block=False)            # get a trained local model from the queue
            flag = global_model.Incre_FedAvg(w_in=local_model)  # add it to partial model
            if flag == 1:
                done.set()                                               # if enough number of local models are added to partial model
                break                                                   # this process can be shut down
        else: 
            time.sleep(1)                                               # if the queue is empty, keep scaning

def main():
    config = Config().parse_args()
    mp.set_start_method('spawn', force=True)
    # initialize global model
    global_model = initialize_models(config.num_gpu, config.num_local_models_per_gpu, config)
    coordinator = clients_coordinator(clients_list = list(range(int(config.num_users))), 
                    num_of_gpus = config.num_gpu)   

    
    queue = mp.Queue(maxsize=2)
    GPU_Containers = []
    for gpu_idx, users in coordinator.items():
        GPU_Containers.append(GPU_Container(users = users, \
                                           device = torch.device('cuda:'+str(gpu_idx)), \
                                           config=config, queue=queue))

    for i in range(config.num_steps):
        done = mp.Event()
        print("Federated Step: ", i)
        # pool = mp.Pool()
        assert len(GPU_Containers) == config.num_gpu
        local_process_list = []
        for gpu_launcher in GPU_Containers:
            gpu_launcher.update_done(done)
            gpu_launcher.update_true_global(global_model)
            local_process_list += gpu_launcher.launch_gpu()

        
        # partial_model_process = mp.Process(launch_process_update_partial, \
        #                         args=(queue, global_model))  
        # partial_model_process.start()
        launch_process_update_partial(queue, global_model, done)


        for p in local_process_list:
            p.join()
        # partial_model_process.join()

    save_checkpoint(global_model.saved_state_dict, 'checkpoint_global.pth')

        

if __name__ == '__main__':
    main()



