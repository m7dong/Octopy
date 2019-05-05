import torch

class Global_Model:
    def __init__(self, state_dict, capacity, num_users):
        '''
        capacity = num of gpus
        '''
        self.incre_counter = 0      # within current round, how many partial global you have received
                                    # NOTE: when you pull global model, you may want to make sure that incre_conuter == 0
        self.round = 0              # current round of Federated Learning
        self.capacity = capacity    # how many partial global you expect to receive per round
        self.state_dict = state_dict  # weights of global model
        self.saved_state_dict = state_dict
        self.num_users = num_users


    def Incre_FedAvg(self, w_in):
        if self.incre_counter == 0:
            for k in self.state_dict.keys():
                self.state_dict[k] = torch.div(w_in[k], self.num_users)
            self.incre_counter += 1
            return 0

        for k in self.state_dict.keys():  # iterate every weight element
            self.state_dict[k] += torch.div(w_in[k], self.num_users)

        self.incre_counter += 1
        if self.incre_counter == self.capacity:
            print('This is the end of this round ...')
            self.round += 1
            self.incre_counter = 0
            self.saved_state_dict = self.state_dict
            return 1

        return 0



        
    
