import torch
from rl_games.algos_torch import torch_ext

def safe_load(filename):
    return torch_ext.safe_filesystem_op(torch.load, filename, map_location=torch.device('cpu'))

def load_checkpoint(filename):
    print("=> loading checkpoint '{}' on cpu.".format(filename))
    state = safe_load(filename)
    return state