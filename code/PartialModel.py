import torch
from Lenet import Net
from multiprocessing import Process, Lock
import copy
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Partial_Model:
    def __init__(self, device, capacity, global_model):
        """
        capacity = num of local models
        """
        self.device = device #which gpu this Partial_Model is located
        self.state_dict = copy.deepcopy(global_model.state_dict) # weights of partial model
        self.capacity = capacity # how many local models specified in the same GPU
        self.state_dict.to(self.device)  # TODO: it will return an error.
        self.counter = 0
        
    def partial_updates_sum(self, w_in):
        # w_in represents weights from a local model
        for k in self.state_dict.keys(): # iterate every weight element
            self.state_dict[k] += w_in[k]
        self.counter += 1
        if self.counter == self.capacity:
            # 1. divide
            for k in self.state_dict.keys():
                self.state_dict[k] /= self.counter
        return self.state_dict


def update_partial_sum(l, user_list_, partial_model_):
    l.acquire()
    try:
        for i in user_list_:
            partial_model_.partial_updates_sum(i)
    finally:
        l.release()

if __name__ == '__main__':
    # Note: test codes
    # number of local = k
    
    from Lenet import Net
    from multiprocessing import Process, Lock
    
    #init_dict = torch.zeros(5, 3, dtype=torch.long)
    number_of_local = k
    gpu = 1
    partial_model=Partial_Model(gpu_index = gpu, capacity = num_of_local, global_model = global_model)
    lock = Lock()
    for i in range(k):
        #read in user
        #....
        #.....
    #to avoid too many times of initializing processes, only generates k/10 processes.
    for num in range(k/10):
        Process(target=update_partial_sum, args=(lock, user_list_k, partial_model)).start()

    for k in partial_model.state_dict.keys():
        partial_model.local_model[k] = torch.div(partial_model.state_dict[k], partial_model.capacity)


