import torch



def load_checkpoint(snapshot_path, num_gpu=1):
    if isinstance(snapshot_path, str):
        cp = torch.load(snapshot_path)
        if type(cp) is not dict:
            # model -> state_dict
            cp={'epoch':0, 'state_dict': cp.state_dict()}
    
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





