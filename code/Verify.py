import torch



def convert_state_dict(state_dict, num_gpu):
    # multi-single gpu conversion
    if num_gpu==1 and list(state_dict.keys())[0][:7]=='module.':
        # modify the saved model for single GPU
        tobe_popped = []
        # Should not modify state_dict during iteration
        # otherwise will have erderedDict mutated during iteration
        for k,v in state_dict.items():
            tobe_popped.append(k)
        for k in tobe_popped:
            state_dict[k[7:]] = state_dict[k]
            state_dict.pop(k,None)            
    elif num_gpu>1 and (len(list(state_dict.keys())[0])<7 or list(state_dict.keys())[0][:7]!='module.'):
        # modify the single gpu model for multi-GPU
        tobe_popped = []
        for k,v in state_dict.items():
            tobe_popped.append(k)
        for k in tobe_popped:
            state_dict['module.'+k] = state_dict[k]
            state_dict.pop(k,None)


def load_checkpoint(snapshot_path, num_gpu=1):
    if isinstance(snapshot_path, str):
        cp = torch.load(snapshot_path)
        if type(cp) is not dict:
            # model -> state_dict
            cp={'epoch':0, 'state_dict': cp.state_dict()}
    else:
        cp={'epoch':0, 'state_dict': snapshot_path}
    convert_state_dict(cp['state_dict'], num_gpu)
    return cp


def get_avg_state_dict(paths):
	avg_state_dict = None
	for idx, path in enumerate(paths):
		model_state_dict = load_checkpoint(path)
		if idx == 0:
			avg_state_dict = model_state_dict
		else:
			for k,v in avg_state_dict.items():
				avg_state_dict[k] += model_state_dict[k] / len(paths)

	return avg_state_dict


if __name__ == '__main__':
	paths = []

    main(paths)





