#local2partial within the same GPU
import torch

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            
    if __name__ == '__main__':
        # Note: test codes
        # one local
        from Lenet import Net1, Net2, Net3
        model_list = [Net1, Net2, Net3]
        num_of_local = len(model_list)
        #initialize
        init_dict = torch.zeros(5, 3, dtype=torch.long)
        partial_model = Partial_Model(state_dict = init_dict, capacity = num_of_local)
        for i in range(len(model_list)):
            net = model_list[i]()
            partial_model.partial_updates_sum(net.state_dict)
        #calculate avg
        for k in partial_model.state_dict.keys():
            partial_model.state_dict[k] = torch.div(partial_model.state_dict[k], partial_model.capacity)

        #update to global model
