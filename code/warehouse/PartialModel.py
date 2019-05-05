import torch
from Lenet import Net
from multiprocessing import Process, Lock
import copy
from warehouse.funcs import *
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Partial_Model:
    def __init__(self, device, capacity, global_model):
        """
        capacity = num of local models
        """
        self.device = device #which gpu this Partial_Model is located
        self.state_dict = move_to_device(copy.deepcopy(global_model.state_dict), device) # weights of partial model
        for k in self.state_dict.keys(): # iterate every weight element
            self.state_dict[k] = 0
        self.capacity = capacity # how many local models specified in the same GPU
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
            return 1  # return a flag

        return 0
