from PartialModel import Partial_Model 
from warehouse.funcs import *
from Users import User
import models
import torch.multiprocessing as mp
import copy 
import time
import torch

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


    def step_training(self, processing_index, step):
        print("launch local model training process: ", device, processing_index)
        
        ready_model = models.__dict__[self.config.model]()
        ready_model.to(self.device).load_state_dict(self.true_global)
        for user_index in self.user_list_for_processings[processing_index]:
            ready_model.load_state_dict(self.true_global)
            current_user = User(user_index=user_index, ready_model=ready_model, local_epoch=self.config.num_epochs)
            current_user.local_train(step, self.config.num_steps)
            self.local_model_queue.put(copy.deepcopy(current_user.net.state_dict()), block=True)
        
        print("Ending local model training process: ", self.device, processing_index)
        self.done.clear()


    def launch_one_training_process(self, gpu_index, processing_index):
        processing_index_global = (self.config.num_local_models_per_gpu + 1) * gpu_index + processing_index
        
        while True:
            if self.flags[processing_index_global] < config.num_steps:
                print(gpu_index, processing_index, self.flags[processing_index_global])
                step = int(self.flags[processing_index_global].data.tolist()) + 1 
                self.step_training(processing_index, step)
                self.flags[processing_index_global] += 1
                done.wait()
            else:
                break
    

    def launch_partial_update_process(self, gpu_index, processing_index):
        processing_index_global = (self.config.num_local_models_per_gpu + 1) * gpu_index + processing_index
        while True:   # scan the queue
            if not self.local_model_queue.empty():
                local_model = self.local_model_queue.get(block=False)            # get a trained local model from the queue
                flag = self.partial_model.partial_updates_sum(w_in=local_model)  # add it to partial model
                if flag == 1:
                    self.flags[processing_index_global] += 1                     # if enough number of local models are added to partial model
                    self.global_queue.put(move_to_device(copy.deepcopy(self.partial_model.state_dict),
                                  torch.device('cpu')), block=True)
                    self.done.wait()
                    self.partial_model.counter = 0
            else: 
                time.sleep(1)  
            if self.flags[processing_index_global] >= self.config.num_steps:
                break
        self.done.wait()
        #print("close partial model updating process")


    def launch_gpu(self):
        assert self.done is not None

        local_process_list = []
        gpu_index = int(str(self.device)[5:])
        for processing_index in range(self.gpu_parallel):
            new_p = mp.Process(target=self.launch_one_training_process, args=(gpu_index, processing_index))
            new_p.start()
            local_process_list.append(new_p)

        global_p = mp.Process(target=self.launch_partial_update_process, \
                    args=(gpu_index, self.gpu_parallel))
        global_p.start()
        local_process_list.append(global_p) 


        return local_process_list        

        

        
