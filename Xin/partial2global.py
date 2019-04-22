
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
        
def clients_coordinator(clients_list, num_of_gpus):
    '''
    Input: 
        clients_list: list of clients' index to train
        num_of_gpus: how many gpus we can use to train
    Output:
        Dict: key is index of gpu, value is clients' index for this gpu.
    '''
    coordinator = {}
    splited_clients_list = chunkIt(clients_list, num_of_gpus)
    for i in range(num_of_gpus):
        coordinator[i] = splited_clients_list[i]
    return coordinator


def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

if __name__ == '__main__':
    # NOTE: test codes
    from Lenet import Net
    net = Net()
    num_of_gpus = 8
    global_model = Global_Model(state_dict = net.state_dict, capacity = num_of_gpus)
    coordinator = clients_coordinator(clients_list = list(range(int(20))), num_of_gpus = num_of_gpus)
    # NOTE: Once partial global on i-th device processed len(coordinator[i]) local clients,
    #       it can call global_model.Incre_FedAvg(partial_global's state_dict)