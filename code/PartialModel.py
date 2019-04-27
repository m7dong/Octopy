#local2partial within the same GPU
import torch
from Lenet import Net

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Partial_Model:
    def __init__(self, state_dict, capacity, global_model):
        """
        capacity = num of local models
        """
        self.capacity = capacity # how many local models specified in the same GPU
        self.state_dict = state_dict # weights of partial model
        self.global_model = global_model
        self.true_global = self.global_model.state_dict

        self.local_model = Net()
        self.local_model_list = [self.local_model] * self.capacity

    def partial_updates_sum(self, w_in):
        #w_in represents weights from a local model
        for k in self.state_dict.keys(): # iterate every weight element
            self.state_dict[k] += w_in[k]
        
        return self.state_dict

    def pull_global(self):
        assert self.global_model.incre_counter == 0   # make sure that global is the true global
        self.true_global = self.global_model.state_dict

        
        
        
        
#local2partial within the same GPU
import torch
from multiprocessing import Process, Lock


class Partial_Model:
    def __init__(self, state_dict, capacity):
        """
        capacity = num of local models
        """
        self.capacity = capacity # how many local models specified in the same GPU
        self.state_dict = state_dict # weights of partial model


    def partial_updates_sum(self, w_in):
        #w_in represents weights from a local model
        for k in self.state_dict.keys(): # iterate every weight element
            self.state_dict[k] += w_in[k]
        
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
    
    init_dict = torch.zeros(5, 3, dtype=torch.long)
    partial_model=Partial_Model(state_dict = init_dict, capacity = num_of_local)
    number_of_local = k
    model_list=[]
    lock = Lock()
    for i in range(k):
        #read in user
        #....
        #.....
    #to avoid times of initializing processes, only generates k/10 processes.
    for num in range(k/10):
        Process(target=update_partial_sum, args=(lock, user_list, partial_model)).start()

    for k in partial_model.state_dict.keys():
        partial_model.state_dict[k] = torch.div(partial_model.state_dict[k], partial_model.capacity)

#update to global model

