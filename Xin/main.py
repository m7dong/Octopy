from partial2global import Global_Model
from local2partial import Partial_Model
from Lenet import Net
from funcs import *

import torch

def initialize_models(num_of_gpus):
	local_net1, local_net2, local_net3 = Net, Net, Net
    model_list = [local_net1, local_net2, local_net3]
    num_of_local = len(model_list)

    # initialize global model on CPU
    global_net = Net()
    global_model = Global_Model(state_dict = global_net.state_dict, capacity = num_of_gpus)
    
    # NOTE: Once partial global on i-th device processed len(coordinator[i]) local clients,
    #       it can call global_model.Incre_FedAvg(partial_global's state_dict)
    
    # Note: test codes
    # one local
    # initialize partial models on GPU
    init_dict = torch.zeros(5, 3, dtype=torch.long)
    partial_model = Partial_Model(state_dict = init_dict, capacity = num_of_local, global_model=global_model)
    for i in range(len(model_list)):
        net = model_list[i]()
        #partial_model.partial_updates_sum(net.state_dict)

    return global_model, partial_model


def main():
	# load data
	dataset_train, dataset_test = get_dataloader()

	# initialize models
	num_of_gpus = 8
	global_model, partial_model = initialize_models(num_of_gpus)
	coordinator = clients_coordinator(clients_list = list(range(int(20))), num_of_gpus = num_of_gpus)	

    total_rounds = 10
    for t in range(total_rounds):
    	# pull global to true global on each GPU
    	if t > 0:
    		while global_model.incre_counter != 0:
    			continue
    		partial_model.pull_global()

    	# step training
    	launch_training_on_different_gpu()
    
	    # calculate avg
	    for k in partial_model.state_dict.keys():
	        partial_model.state_dict[k] = torch.div(partial_model.state_dict[k], partial_model.capacity)

	    # update global model on CPU
	    global_model.Incre_FedAvg(partial_global.state_dict)




if __name__ == '__main__':
	main()



