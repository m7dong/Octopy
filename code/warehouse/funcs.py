'''
    funcs.py
'''
import torch
from collections import OrderedDict
import time

def get_dataloader():
    from torchvision import datasets, transforms
    trans_mnist = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST(
        './data/MNIST/', train=True, download=False, transform=trans_mnist)
    dataset_test = datasets.MNIST(
        './data/MNIST/', train=False, download=False, transform=trans_mnist)
    return dataset_train, dataset_test




def launch_process_update_partial(local_model_queue, global_model, done):
    while True:   # scan the queue
        if not local_model_queue.empty():  
            local_model = local_model_queue.get(block=False)            # get a trained local model from the queue
            flag = global_model.Incre_FedAvg(w_in=local_model)  # add it to partial model
            if flag == 1:
                done.set()                                               # if enough number of local models are added to partial model
                break                                                   # this process can be shut down
        else: 
            time.sleep(1)                                               # if the queue is empty, keep scaning

def gpu_update_users(user_list, gpu_list):
    coordinator = clients_coordinator(clients_list = user_list, 
                    num_of_gpus = len(gpu_list))
    for gpu_idx, users in coordinator.items():
        gpu_list[gpu_idx].update_users(users)  

    return gpu_list





def move_to_device(state_dict, target_device):
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(target_device)
    return state_dict 


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


def save_checkpoint(model, filename='checkpoint.pth', optimizer=None, epoch=1):
    # model or model_state
    out = model if type(model) is OrderedDict else model.state_dict()
    out = move_to_device(out, torch.device('cpu'))
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': out,
            'optimizer' : optimizer.state_dict()
        }, filename)


  
