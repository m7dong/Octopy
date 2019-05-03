import torch
import torch.multiprocessing as mp

from GlobalModel import Global_Model
# from PartialModel import Partial_Model
from warehouse.funcs import *
from GPUContainer import GPU_Container
from Config import Config
import models
import time


def initialize_global_model(config):
    # initialize global model on CPU
    global_net = models.__dict__[config.model]()
    #global_model = Global_Model(state_dict = global_net.state_dict(), capacity = config.num_users)
    global_model = Global_Model(state_dict = global_net.state_dict(), capacity = config.num_gpu)
    return global_model


def main():
    config = Config().parse_args()
    mp.set_start_method('spawn', force=True)

    # initialize global model
    global_model = initialize_global_model(config)
 
    # setup queue for trained local models
    queue = mp.Queue(maxsize=2)

    flags = torch.zeros((config.num_local_models_per_gpu+1) * config.num_gpu)
    for f in flags:
        f.share_memory_()

    # setup gpu container for each gpu
    GPU_Containers = []
    for gpu_idx in range(config.num_gpu):
        GPU_Containers.append(GPU_Container(device = torch.device('cuda:'+str(gpu_idx)), \
                                           config=config, queue=queue, flags=flags))

    done = mp.Event()              # setup up event for queue
    assert len(GPU_Containers) == config.num_gpu
    
    # specify the users-list for this round
    GPU_Containers = gpu_update_users(user_list = list(range(int(config.num_users))), gpu_list = GPU_Containers)  
    local_process_list = []
    
    # start multiprocessing training for each gpu
    for gpu_launcher in GPU_Containers:
        gpu_launcher.update_done(done)   # update event for each round
        gpu_launcher.update_true_global(global_model)   #update global model for each round
        local_process_list += gpu_launcher.launch_gpu()

    i = 0
    while True:    
        if int(flags.sum().data.tolist()) == (config.num_local_models_per_gpu+1) * (i+1):
            #print("gathering result iter ", i)
            launch_process_update_partial(queue, global_model, done)
            for gpu_launcher in GPU_Containers:
                gpu_launcher.update_true_global(global_model)
            i += 1
            done.set()  
            if i > config.num_steps:
                break

    for p in local_process_list:
        p.join()

    # save_checkpoint(global_model.saved_state_dict, 'checkpoint_global.pth')

        

if __name__ == '__main__':
    main()



