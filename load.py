import torch
from torch_geometric.utils import to_undirected

model_dir = '/export/home/phys/kyungmip/L1DeepMETv2/ckpts_sep21/'       # name of the ckpts folder'

loaded_model = torch.jit.load(f'{model_dir}/scripted_model.pt')

print(loaded_model.eval())

