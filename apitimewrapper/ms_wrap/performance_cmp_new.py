import mindspore as ms
import mindspore
from mindspore import Tensor, ops, nn
import torch
import numpy as np
import time


class Net(nn.Cell):
    def __init__(self, func):
        super(Net, self).__init__()
        self.func = func

    def construct(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class torchNet(torch.nn.Module):
    def __init__(self, func):
        super(torchNet, self).__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# ms.set_context(mode=ms.GRAPH_MODE)
np_x = np.random.randn(200, 20, 20, 20).astype(np.float32)

args = (np_x,)
kwargs = {}
ms_func_name = ops.rrelu
torch_func_name = torch.rrelu


def performance_cmp(ms_func_name, torch_func_name, run_time, performance_multiple, *args, **kwargs):
    ms_args = []
    torch_args = []

    for arg in args:
        if isinstance(arg, np.ndarray):
            ms_arg = Tensor(arg)
            torch_arg = torch.tensor(arg)
            ms_args.append(ms_arg)
            torch_args.append(torch_arg)
        else:
            ms_args.append(arg)
            torch_args.append(arg)
    net = Net(ms_func_name)
    net(*ms_args, **kwargs)
    tnet = torchNet(torch_func_name)
    tnet(*torch_args, **kwargs)

    ms_start_time = time.perf_counter()

    for i in range(run_time):
        net(*ms_args, **kwargs)

    ms_end_time = time.perf_counter()
    mindspore_cost_time = (ms_end_time - ms_start_time) / run_time
    print(f"mindspore {ms_func_name.__name__} run {run_time} times cost time: {mindspore_cost_time}.")

    torch_start_time = time.perf_counter()
    for i in range(run_time):
        tnet(*torch_args, **kwargs)
    torch_end_time = time.perf_counter()
    torch_cost_time = (torch_end_time - torch_start_time) / run_time
    print(f"torch {torch_func_name.__name__} run {run_time} times cost time: {torch_cost_time}.")
    print(mindspore_cost_time / torch_cost_time)


performance_cmp(ms_func_name, torch_func_name, 10, 2, *args, **kwargs)