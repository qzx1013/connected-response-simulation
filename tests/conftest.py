import os

os.environ["TANGENT_BP_BOUNDED_2X2_TREE_CACHE"] = ""

import torch

torch.set_num_threads(2)
torch.set_grad_enabled(False)
