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
		model_state_dict = load_checkpoint(path)['state_dict']
		if idx == 0:
			avg_state_dict = model_state_dict 
			for k,v in avg_state_dict.items():
				avg_state_dict[k] = model_state_dict[k] / len(paths)
		else: 
			for k,v in avg_state_dict.items():
				avg_state_dict[k] += model_state_dict[k] / len(paths)

	return avg_state_dict


if __name__ == '__main__':
	paths = ['checkpoint_%d.pth' % user_index for user_index in range(10)]
	true = get_avg_state_dict(paths)
	global_path = 'checkpoint_global.pth'
	out = load_checkpoint(global_path)['state_dict']
	for k, i in true.items():
		# if (true[k] == out[k]):
		# 	print(1)
		# else:
		print('true: ', true[k])
		print('out: ', out[k])





