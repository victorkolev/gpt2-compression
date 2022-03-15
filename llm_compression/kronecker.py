import torch
import numpy as np
import torch.nn as nn
import transformers


def get_kronecker_decomposition(A, Bshape):
    blocks = map(
        lambda blockcol: torch.chunk(blockcol, Bshape[0], 0),
        torch.chunk(A, Bshape[1], 1),
    )
    Atilde = torch.vstack(
        [torch.ravel(block) for blockcol in blocks for block in blockcol]
    )
    U, s, V = torch.svd(Atilde)
    V = V.t()
    Cshape = A.shape[0] // Bshape[0], A.shape[1] // Bshape[1]
    idx = torch.argmax(s)
    B = torch.sqrt(s[idx]) * U[:, idx].reshape(Bshape)
    C = torch.sqrt(s[idx]) * V[idx, :].reshape(Cshape)
    return B.contiguous(), C.contiguous()


class KroneckerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for mod in self.model.modules():
            if hasattr(mod, "weight") and mod.weight.dim() == 2:
                self._kr_decompose_weight(mod)

    def _kr_decompose_weight(self, mod):
        if mod.weight.shape[0] % 2 == 0:
            bshape = (mod.weight.shape[0] // 2, mod.weight.shape[1])
        else:
            bshape = (mod.weight.shape[0], mod.weight.shape[1] // 2 )

        a, b = get_kronecker_decomposition(mod.weight, bshape)

        mod.register_parameter("kr_A", nn.Parameter(a))
        mod.register_parameter("kr_B", nn.Parameter(b))
        with torch.no_grad():
            mod.weight.copy_(torch.kron(mod.kr_A, mod.kr_B))

    def compute_kr_gradient(self):
        for mod in self.model.modules():
            if hasattr(mod, "kr_A"):
                with torch.no_grad():
                    g = mod.weight.grad
                    kr_a_grad = torch.zeros_like(mod.kr_A)
                    kr_b_grad = torch.zeros_like(mod.kr_B)

                    for i in range(mod.kr_A.shape[0]):
                        for j in range(mod.kr_A.shape[1]):
                            kr_a_grad[i, j] = torch.sum(
                                g[
                                    i * mod.kr_B.size(0) : (i + 1) * mod.kr_B.size(0),
                                    j * mod.kr_B.size(1) : (j + 1) * mod.kr_B.size(1),
                                ]
                                * mod.kr_B
                            )

                    for i in range(mod.kr_B.shape[0]):
                        for j in range(mod.kr_B.shape[1]):
                            kr_b_grad[i, j] = torch.sum(
                                g[i :: mod.kr_B.shape[0], j :: mod.kr_B.shape[1]]
                                * mod.kr_A
                            )

                    mod.kr_A.grad = kr_a_grad
                    mod.kr_B.grad = kr_b_grad

    def parameters(self):
        for n, p in self.model.named_parameters():
            if "kr_" in n or p.dim() != 2:
                yield p

    def named_parameters(self):
        for n, p in self.model.named_parameters():
            n = "model." + n
            if "kr_" in n or p.dim() != 2:
                yield n, p

    def forward(self, *inputs):
        self.recalculate_weights()
        return self.model(*inputs)

    def recalculate_weights(self):
        self.model.zero_grad()
        for n, mod in self.model.named_modules():
            if hasattr(mod, "weight") and mod.weight.dim() == 2:
                with torch.no_grad():
                    mod.weight.copy_(torch.kron(mod.kr_A, mod.kr_B))



def model_decompose_kronecker(model):
    for n, mod in model.named_modules():
        if hasattr(mod, "weight") and mod.weight.dim() == 2:
            if mod.weight.shape[0] % 2 == 0:
                bshape = (mod.weight.shape[0] // 2, mod.weight.shape[1])
            else:
                bshape = (mod.weight.shape[0], mod.weight.shape[1] // 2 )

            a, b = get_kronecker_decomposition(mod.weight, bshape)

            mod.register_parameter("kr_A", nn.Parameter(a))
            mod.register_parameter("kr_B", nn.Parameter(b))
            # model._parameters.pop(n+".weight")
            with torch.no_grad():
                mod.weight.copy_(torch.kron(mod.kr_A, mod.kr_B))


def recalculate_weights(model):
    model.zero_grad()
    for n, mod in self.model.named_modules():
        if hasattr(mod, "weight") and mod.weight.dim() == 2:
            with torch.no_grad():
                mod.weight.copy_(torch.kron(mod.kr_A, mod.kr_B))


def compute_kr_gradient(model):
    for mod in model.modules():
        if hasattr(mod, "kr_A"):
            with torch.no_grad():
                g = mod.weight.grad
                kr_a_grad = torch.zeros_like(mod.kr_A)
                kr_b_grad = torch.zeros_like(mod.kr_B)

                for i in range(mod.kr_A.shape[0]):
                    for j in range(mod.kr_A.shape[1]):
                        kr_a_grad[i, j] = torch.sum(
                            g[
                                i * mod.kr_B.size(0) : (i + 1) * mod.kr_B.size(0),
                                j * mod.kr_B.size(1) : (j + 1) * mod.kr_B.size(1),
                            ]
                            * mod.kr_B
                        )

                for i in range(mod.kr_B.shape[0]):
                    for j in range(mod.kr_B.shape[1]):
                        kr_b_grad[i, j] = torch.sum(
                            g[i :: mod.kr_B.shape[0], j :: mod.kr_B.shape[1]]
                            * mod.kr_A
                        )

                mod.kr_A.grad = kr_a_grad
                mod.kr_B.grad = kr_b_grad


def apply_kronecker_to_model(model):
    model_decompose_kronecker(model)
    model.compute_kr_gradient = compute_kr_gradient.__get__(model)
    model.recalculate_weights = recalculate_weights.__get__(model)

    def parameters(model):
        for n, p in model.named_parameters():
            if "kr_" in n:
                yield p

    model.parameters = parameters.__get__(model)


from functools import wraps
class HfKroneckerCallback(transformers.TrainerCallback):

    def on_init_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        apply_kronecker_to_model(model)

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        optimizer._step = optimizer.step

        @wraps(optimizer._step)
        def step(optimizer):
            model.compute_kr_gradient()
            optimizer._step()
            model.recalculate_weights()

        optimizer.step = step.__get__(optimizer)






if __name__ == "__main__":
    # for _ in range(100):
    #     m = torch.rand(20, 20) 
    #     a, b = get_kronecker_decomposition(m, (2, 1))
    #     aa, bb = kr_np(m.numpy(), (2, 1))
    #     d = lambda x, y: (x - y).norm().item()
    #     kr = torch.kron(a, b)
    #     krnp = np.kron(aa, bb)
    #     # ipdb.set_trace()
    #     print(d(torch.tensor(krnp).reshape(*kr.shape), kr))
    net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    # net = nn.Linear(2, 2)
    net = Kronecker(net)
    # ipdb.set_trace()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = shady_optimizer(optimizer, net)
    # optimizer = optim.SGD(net.model.parameters(), lr=0.01, momentum=0.9)

    for _ in range(100):
        x = torch.rand(10, 2)
        y = x**2
        # ipdb.set_trace()
        preds = net(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
