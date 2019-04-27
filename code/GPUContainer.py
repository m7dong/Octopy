from PartialModel import Partial_Model 
from funcs import chunkIt
from Users import User
from Lenet import Net

class GPU_Container:
    def __init__(self, users, global_model, gpu_parallel, device):
        self.users = users
    	self.partial_model = Partial_Model(capacity = len(self.users), global_model = global_model, device = device)
        self.gpu_parallel = gpu_parallel
        self.device = device
        
        self.split_for_processing()
        
    def split_for_processings(self):
        self.user_list_for_processings = chunkIt(self.users, self.gpu_parallel)
        
    def lunch_one_processing(self, processing_index):
        ready_model = Net().load_state_dict(self.partial_model.true_global).to(self.device)
        for user_index in self.user_list_for_processings[processing_index]:
            current_user = User(user_index=user_index, ready_model=ready_model)
            current_user.local_train()
            #TODO: how to push local model (subprocessing N) to partial global (main processing)
        
