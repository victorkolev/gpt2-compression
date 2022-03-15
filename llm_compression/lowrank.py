import torch
import torch.nn as nn

from transformers import TrainerCallback

def decrease_rank(model):
    for n, mod in model.named_modules():
        if hasattr(mod, "weight") and mod.weight.dim() == 2:
            if not hasattr(mod, "rank"):
                setattr(mod, "rank", min(*mod.weight.shape))
            mod.rank -= int(mod.rank * model.rank_decrease)
            U, s, V = torch.svd(mod.weight)
            V = V.t()
            with torch.no_grad():
                mod.weight.copy_(U[:, :mod.rank] @ torch.diag(s[:mod.rank]) @ V[:mod.rank, :])


from functools import wraps
class HfSVDCallback(TrainerCallback):

    def on_init_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        setattr(model, "rank_decrease", 0.01)
        model.decrease_rank = decrease_rank.__get__(model)

    def on_epoch_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        model.decrease_rank()
        print("Rank decreased")



