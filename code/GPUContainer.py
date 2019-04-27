from PartialModel import Partial_Model 


class GPU_Container:
    def __init__(self, partial_model, num_users):
    	self.partial_model = partial_model
    	self.users = []