from PartialModel import Partial_Model 
from warehouse.funcs import *
from Users import User
from Lenet import Net
import torch.multiprocessing as mp
import copy 
import time
import torch

def launch_one_processing(processing_index, true_global, device, 
                            user_list_for_processings, local_model_queue, config, done):
    print("launch local model training process: ", device, processing_index)
    #print('true global', true_global)
    ready_model = Net()
    ready_model.to(device).load_state_dict(true_global)
    #print('1')
    for user_index in user_list_for_processings[processing_index]:
        #print('2: ', ready_model, '|', true_global['fc2.bias'].device)
        ready_model.load_state_dict(true_global)
        #print('3')
        current_user = User(user_index=user_index, ready_model=ready_model, local_epoch=config.num_epochs)
        #print('4')
        current_user.local_train()
        #print('5')
        local_model_queue.put(move_to_device(copy.deepcopy(current_user.net.state_dict()), torch.device('cpu')), block=True)
    print("Ending local model training process: ", device, processing_index)
    done.wait()
    print("**Ending local model training process: ", device, processing_index)





class GPU_Container:
    def __init__(self, users, global_model, device, config, queue):
        self.users = users
        self.gpu_parallel = config.num_local_models_per_gpu
        self.device = device
        #m = mp.Manager()
        self.local_model_queue = queue
        
        self.split_for_processings()
        self.global_model = global_model
        #self.true_global = global_model.state_dict.to(self.device)
        self.true_global = move_to_device(copy.deepcopy(global_model.state_dict), self.device)
        self.config = config
        self.done = None
        

    def split_for_processings(self):
        self.user_list_for_processings = chunkIt(self.users, self.gpu_parallel)
        

    def update_users(self, users):
        self.users = users
        self.split_for_processings()

    def update_done(self, done):
        self.done = done
            

    def launch_gpu(self):
        assert self.done is not None
        
        local_process_list = []
        for processing_index in range(self.gpu_parallel):
            new_p = mp.Process(target=launch_one_processing, \
                    args=(processing_index, self.true_global, self.device, self.user_list_for_processings,\
                            self.local_model_queue, self.config, self.done))
            new_p.start()
            local_process_list.append(new_p)


        return local_process_list

        

        
