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
    #print('true global', true_global)
    ready_model = models.__dict__[config.model]()
    ready_model.to(device).load_state_dict(true_global)
    for user_index in user_list_for_processings[processing_index]:
        ready_model.load_state_dict(true_global)
        current_user = User(user_index=user_index, ready_model=ready_model, local_epoch=config.num_epochs)
        current_user.local_train(step, config.num_steps)
        #print("Ending local model training for user: ", device, processing_index, user_index)
        local_model_queue.put(copy.deepcopy(current_user.net.state_dict()), block=True)
        #print("result in local queue for user: ", device, processing_index, user_index)
    print("Ending local model training process: ", device, processing_index)
    print("**Ending local model training process: ", device, processing_index)
    done.clear()

def launch_one_training_process(gpu_index, processing_index, true_global, device, 
                            user_list_for_processings, local_model_queue, config, done, flags):
    processing_index_global = (config.num_local_models_per_gpu + 1) * gpu_index + processing_index
    while True:
        if flags[processing_index_global] < config.num_steps:
            print(gpu_index, processing_index, flags[processing_index_global])
            step = int(flags[processing_index_global].data.tolist()) + 1 
            step_training(processing_index, true_global, device, 
                            user_list_for_processings, local_model_queue, config, done, step)
            flags[processing_index_global] += 1
            #print('wait', gpu_index, processing_index)
            #print(flags[processing_index_global])
            done.wait()
            #print('resume', gpu_index, processing_index)
        else:
            break
    #print("close training process", gpu_index, processing_index)
    

def launch_partial_update_process(gpu_index, processing_index, global_queue, local_model_queue, partial_model, config, done, flags):
    processing_index_global = (config.num_local_models_per_gpu + 1) * gpu_index + processing_index
    while True:   # scan the queue
        if not local_model_queue.empty():
            #print("Processing local queue")  
            local_model = local_model_queue.get(block=False)            # get a trained local model from the queue
            flag = partial_model.partial_updates_sum(w_in=local_model)  # add it to partial model
            if flag == 1:
                flags[processing_index_global] += 1                     # if enough number of local models are added to partial model
                global_queue.put(move_to_device(copy.deepcopy(partial_model.state_dict),
                              torch.device('cpu')), block=True)
                #print('wait', gpu_index, processing_index)
                #print(flags[processing_index_global])
                done.wait()
                #print('resume', gpu_index, processing_index)
                partial_model.counter = 0
        else: 
            time.sleep(1)  
        if flags[processing_index_global] >= config.num_steps:
            break
    done.wait()
    #print("close partial model updating process")


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
        self.partial_model = Partial_Model(self.device, len(self.user_list_for_processings), self.true_global, self.config)


    def launch_gpu(self):
        assert self.done is not None

        local_process_list = []
        gpu_index = int(str(self.device)[5:])
        for processing_index in range(self.gpu_parallel):
            new_p = mp.Process(target=launch_one_training_process, \
                    args=(gpu_index, processing_index, self.true_global, self.device, self.user_list_for_processings,\
                            self.local_model_queue, self.config, self.done, self.flags))
            new_p.start()
            local_process_list.append(new_p)

        global_p = mp.Process(target=launch_partial_update_process, \
                    args=(gpu_index, self.gpu_parallel, self.global_queue, self.local_model_queue, self.partial_model, self.config, self.done, self.flags))
        global_p.start()
        local_process_list.append(global_p) 


        return local_process_list        

        

        
