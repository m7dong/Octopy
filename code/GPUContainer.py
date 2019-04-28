from PartialModel import Partial_Model 
from funcs import chunkIt
from Users import User
from Lenet import Net
import torch.multiprocessing as mp
import copy 

def launch_one_processing(processing_index, partial_model, device, 
                            user_list_for_processings, partial_global_queue):
    ready_model = Net().load_state_dict(partial_model.true_global).to(device)
    for user_index in user_list_for_processings[processing_index]:
        ready_model.load_state_dict(partial_model.true_global)
        current_user = User(user_index=user_index, ready_model=ready_model)
        current_user.local_train()
        #TODO: how to push local model (subprocessing N) to partial global (main processing)
        partial_global_queue.put(copy.deepcopy(current_user.net.state_dict()), block=True)

        


class GPU_Container:
    def __init__(self, users, global_model, gpu_parallel, device):
        self.users = users
    	self.partial_model = Partial_Model(capacity = len(self.users), global_model = global_model, device = device)
        self.gpu_parallel = gpu_parallel
        self.device = device
        self.partial_global_queue = mp.Queue(maxsize=2)
        
        self.split_for_processing()
        
    def split_for_processings(self):
        self.user_list_for_processings = chunkIt(self.users, self.gpu_parallel)
            
    def launch_gpu(self, pool):
        for processing_index in range(self.gpu_parallel):
            pool.apply_async(launch_one_processing, \
                    args=(processing_index, self.partial_model, self.device, self.user_list_for_processing,
                            self.partial_global_queue))

if __name__ == '__main__':
    
    pool = mp.Pool()
    for i in range(num_gpus):
        gpu_container = GPU_Container(user_list, global_model, gpu_parallel=4, device=i)
        gpu_container.launch_gpu(pool)
        

        
