import time
import os

import mindspore as ms
from mindspore.common.api import _pynative_executor
import numpy as np
from mindspore import nn
from . import global_param
import yaml
from .my_print import print

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapNNCell = yaml.safe_load(f).get('nn')

NNCell = {}
for f in dir(ms.nn):
    NNCell[f] = getattr(ms.nn, f)


def get_nn_cell():
    global WrapNNCell
    _all_nn_cell = dir(ms.nn)
    return set(WrapNNCell) & set(_all_nn_cell)


def call_decorator(cls, name, parall_execute=False):
    original_call = cls.__call__
    cls.hook_name = 'wrap_' + name

    def new_call(self, *args, **kwargs):
        changed = False
        if global_param.g_stop_hook:
            result = original_call(self, *args, **kwargs)
        else:
            global_param.g_stop_hook = True
            changed = True
            try:
                start_time = time.perf_counter()
                result = original_call(self, *args, **kwargs)
                if not parall_execute:
                    _pynative_executor.sync()
                end_time = time.perf_counter()
                print(f"nn.{self.cls_name} cost_time:{end_time - start_time}")
            except Exception as e:
                raise e
            finally:
                if changed:
                    global_param.g_stop_hook = False
        return result

    cls.__call__ = new_call
    return cls


def wrap_nn_cell_and_bind(parall_execute=False):
    _nn_cell = get_nn_cell()
    for name in _nn_cell:
        call_decorator(NNCell[name], name, parall_execute)


def initialize_hook_nn(parall_execute=False):
    wrap_nn_cell_and_bind(parall_execute)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        conv = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
        conv1 = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        out1 = conv(x)
        out2 = conv1(y)
        return out1, out2


if __name__ == '__main__':
    initialize_hook_nn(parall_execute=False)
    net = Net()
    x = ms.Tensor(np.ones([1, 120, 640]), ms.float32)
    y = ms.Tensor(np.ones([1, 120, 1024, 640]), ms.float32)
    output = net(x, y)
