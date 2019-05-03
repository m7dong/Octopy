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
        self.saved_state_dict = state_dict


    def Incre_FedAvg(self, w_in):
        print('counter: ', self.incre_counter)
        if self.incre_counter == 0:
            self.state_dict = w_in
            self.incre_counter += 1
            print('flag: ', 0)
            return 0

        for k in self.state_dict.keys():  # iterate every weight element
            self.state_dict[k] += torch.div(w_in[k].cpu(), self.incre_counter)
            self.state_dict[k] = torch.div(self.state_dict[k], 1/self.incre_counter+1)

        self.incre_counter += 1
        if self.incre_counter == self.capacity:
            print('This is the end of this round ...')
            self.round += 1
            self.incre_counter = 0
            self.saved_state_dict = self.state_dict
            print('flag: ', 1)
            return 1

        print('flag: ', 0)
        return 0



        
    
