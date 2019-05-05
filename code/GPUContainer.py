from PartialModel import Partial_Model 
from warehouse.funcs import *
from Users import User
import models
import torch.multiprocessing as mp
import copy 
import time
import torch

def step_training(processing_index, true_global, device, 
                            user_list_for_processings, local_model_queue, config, done, step):
    print("launch local model training process: ", device, processing_index)
    ready_model = models.__dict__[config.model]()
    ready_model.to(device).load_state_dict(true_global)
    for user_index in user_list_for_processings[processing_index]:
        ready_model.load_state_dict(true_global)
        current_user = User(user_index=user_index, ready_model=ready_model, local_epoch=config.num_epochs)
        current_user.local_train(step, config.num_steps)
        #print("Ending local model training for user: ", device, processing_index, user_index)
        #print('put: ', current_user.net.state_dict()['fc2.bias'])
        user_model_copy = copy.deepcopy(current_user.net.state_dict())
        local_model_queue.put(user_model_copy, block=True)
        #local_model_queue.put(copy.deepcopy(current_user.net.state_dict()), block=True)
        #time.sleep(3)
        #print(current_user.net.state_dict()['fc2.bias'])
        #time.sleep(3)
        #print("result in local queue for user: ", device, processing_index, user_index)
    print("Ending local model training process: ", device, processing_index)
    print("**Ending local model training process: ", device, processing_index)
    done.clear()

def launch_one_training_process(gpu_index, processing_index, true_global, device, 
                            user_list_for_processings, local_model_queue, config, done, flags):
    processing_index_global = (config.num_local_models_per_gpu) * gpu_index + processing_index
    while True:
        if flags[processing_index_global] < config.num_steps:
            print(gpu_index, processing_index, flags[processing_index_global])
            step = int(flags[processing_index_global].data.tolist()) + 1 
            step_training(processing_index, true_global, device, 
                            user_list_for_processings, local_model_queue, config, done, step)
            flags[processing_index_global] += 1
            #print('wait', gpu_index, processing_index)
            print(flags[processing_index_global])
            done.wait()
            #print('resume', gpu_index, processing_index)
        else:
            break
    print("close training process", gpu_index, processing_index)
    

class GPU_Container:
    def __init__(self, device, config, queue, flags):
        self.users = None
        self.gpu_parallel = config.num_local_models_per_gpu
        self.device = device
        self.global_queue = queue
        self.config = config
        self.done = None
        self.global_model = None

        self.local_model_queue = mp.Queue(maxsize=2)
        self.partial_model = None

        self.flags = flags
        

    def split_for_processings(self):
        self.user_list_for_processings = chunkIt(self.users, self.gpu_parallel)
        

    def update_users(self, users):
        self.users = users
        self.split_for_processings()


    def update_done(self, done):
        self.done = done
    

    def update_true_global(self, global_model):
        self.global_model = global_model
        self.true_global = move_to_device(copy.deepcopy(global_model.saved_state_dict), self.device)
        self.partial_model = Partial_Model(self.device, len(self.users), self.true_global, self.config)


    def launch_gpu(self):
        assert self.done is not None

        local_process_list = []
        gpu_index = int(str(self.device)[5:])
        for processing_index in range(self.gpu_parallel):
            new_p = mp.Process(target=launch_one_training_process, \
                    args=(gpu_index, processing_index, self.true_global, self.device, self.user_list_for_processings,\
                            self.global_queue, self.config, self.done, self.flags))
            new_p.start()
            local_process_list.append(new_p)


        return local_process_list        

        

        
