

from collections import OrderedDict

config = {

	'img_dir' : '/media/ramin/monster/dataset/social/PIPA-relation/imgs/',
	'body_dir' : '/media/ramin/monster/dataset/social/PIPA-relation/all_single_body/',
	'pair_dir' : 'dataset/socialRelation-vision/annotator_consistency3(used in our paper)/',
	'pair_pattern' : 'single_body{0}_{1}_{2}.txt', # pair_num (1,2) -  set (train , test) - 5,16
	'num_class' : 16,
}


def check_weights(state_dict):
	if not any("module" in s for s in state_dict.keys()):
		return state_dict
	
	new_state_dict = OrderedDict()
	del state_dict['fc.weight']
	del state_dict['fc.bias']

	for k, v in state_dict.items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v

	del new_state_dict['fc.bias']
	del new_state_dict['fc.weight']

	return new_state_dict
