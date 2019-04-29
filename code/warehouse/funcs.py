'''
    funcs.py
'''
import torch
from collections import OrderedDict

def get_dataloader():
    from torchvision import datasets, transforms
    trans_mnist = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST(
        './data/MNIST/', train=True, download=False, transform=trans_mnist)
    dataset_test = datasets.MNIST(
        './data/MNIST/', train=False, download=False, transform=trans_mnist)
    return dataset_train, dataset_test


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


  
