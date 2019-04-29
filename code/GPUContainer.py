from PartialModel import Partial_Model 
from warehouse.funcs import chunkIt
from Users import User
from Lenet import Net
import torch.multiprocessing as mp
import copy 
import time

def launch_one_processing(processing_index, true_global, device, 
                            user_list_for_processings, local_model_queue):
    ready_model = Net().load_state_dict(true_global).to(device)
    for user_index in user_list_for_processings[processing_index]:
        ready_model.load_state_dict(true_global)
        current_user = User(user_index=user_index, ready_model=ready_model)
        current_user.local_train()
        local_model_queue.put(copy.deepcopy(current_user.net.state_dict()), block=True)


def launch_process_update_partial(local_model_queue, device, capacity, global_model):
    partial_model = Partial_Model(device=device, capacity=capacity, global_model=global_model)
    while True:   # scan the queue
        if (not self.name_queue.empty()):  
            local_model = local_model_queue.get(block=False)            # get a trained local model from the queue
            flag = partial_model.partial_updates_sum( w_in=local_model) # add it to partial model
            if flag == 1:                                               # if enough number of local models are added to partial model
                break                                                   # this process can be shut down
        else: 
            time.sleep(1)                                               # if the queue is empty, keep scaning
    return partial_model.state_dict

class GPU_Container:
    def __init__(self, users, global_model, gpu_parallel, device):
        self.users = users
        self.gpu_parallel = gpu_parallel
        self.device = device
        self.local_model_queue = mp.Queue(maxsize=2)
        
        self.split_for_processing()
        self.global_model = global_model
        self.true_global = global_model.state_dict.to(self.device)
        

    def split_for_processings(self):
        self.user_list_for_processings = chunkIt(self.users, self.gpu_parallel)
        

    def update_users(self, users):
        self.users = users
        self.split_for_processing()
            

    def launch_gpu(self, pool):
        for processing_index in range(self.gpu_parallel):
            pool.apply_async(launch_one_processing, \
                    args=(processing_index, self.true_global, self.device, self.user_list_for_processing,
                            self.local_model_queue))

        pool.apply_async(launch_process_update_partial, \
                    args=(self.local_model_queue, self.device, len(self.users), self.global_model))        


if __name__ == '__main__':
    
    pool = mp.Pool()
    num_gpus = 2
    for i in range(num_gpus):
        gpu_container = GPU_Container(user_list, global_model, gpu_parallel=4, device=i)
        gpu_container.launch_gpu(pool)
        

        
