
import torch

class Global_Model:
    def __init__(self, state_dict, capacity):
        '''
        capacity = num of gpus
        '''
        self.incre_counter = 0      # within current round, how many partial global you have received
                                    # NOTE: when you pull global model, you may want to make sure that incre_conuter == 0
        self.round = 0              # current round of Federated Learning
        self.capacity = capacity    # how many partial global you expect to receive per round
        self.state_dict = state_dict  # weights of global model


    def Incre_FedAvg(self, w_in):
        for k in self.state_dict.keys():  # iterate every weight element
            self.state_dict[k] += torch.div(w_in[k].cpu(), self.incre_counter)
            self.state_dict[k] = torch.div(self.state_dict[k], 1/self.incre_counter+1)

        self.incre_counter += 1
        self.round += (self.incre_counter / self.capacity)
        self.incre_counter %= self.capacity
        return self.state_dict



        
    
