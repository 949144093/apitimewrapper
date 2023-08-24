import os

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.api import _pynative_executor
import time
from . import global_param
import yaml
from .my_print import print

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapTensorOps = yaml.safe_load(f).get('tensor')

TensorFunc = {}
for f in dir(ms.Tensor):
    TensorFunc[f] = getattr(ms.Tensor, f)


def get_tensor_ops():
    global WrapTensorOps
    _tensor_ops = dir(ms.Tensor)
    return set(WrapTensorOps) & set(_tensor_ops)


class HOOKTensor(object):
    pass


class TensorOPTemplate(nn.Cell):

    def __init__(self, op_name, hook_inside=False, parall_execute=False):
        super(TensorOPTemplate, self).__init__()
        self.op_name_ = op_name
        self.changed_status = hook_inside
        self.parall_execute = parall_execute
        if not global_param.g_stop_hook:
            global_param.g_stop_hook = True
            self.changed_status = True

    def construct(self, *args, **kwargs):
        if self.changed_status:
            start_time = time.perf_counter()
            out = TensorFunc[str(self.op_name_)](*args, **kwargs)
            if not self.parall_execute:
                _pynative_executor.sync()
            end_time = time.perf_counter()
            print(f"Tensor.{self.op_name_} cost_time:{end_time - start_time}")
            self.changed_status = False
            global_param.g_stop_hook = False
        else:
            out = TensorFunc[str(self.op_name_)](*args, **kwargs)
        return out


def wrap_tensor_op(op_name, hook_inside=False, parall_execute=False):
    def tensor_op_template(*args, **kwargs):
        return TensorOPTemplate(op_name, hook_inside, parall_execute)(*args, **kwargs)

    return tensor_op_template


def wrap_tensor_ops_and_bind(hook_inside=False, parall_execute=False):
    _tensor_ops = get_tensor_ops()
    for op_name in _tensor_ops:
        setattr(HOOKTensor, "wrap_" + str(op_name), wrap_tensor_op(op_name, hook_inside, parall_execute))


def initialize_hook_tensor(hook_inside=False, parall_execute=False):
    wrap_tensor_ops_and_bind(hook_inside, parall_execute)
    for attr_name in dir(HOOKTensor):
        if attr_name.startswith("wrap_") and not attr_name.startswith("wrap__") and not isinstance(
                getattr(ms.Tensor, attr_name[5:]), property):
            setattr(ms.Tensor, attr_name[5:], getattr(HOOKTensor, attr_name))
            setattr(ms.common._stub_tensor.StubTensor, attr_name[5:], getattr(HOOKTensor, attr_name))


if __name__ == '__main__':
    initialize_hook_tensor(hook_inside=False, parall_execute=False)
    x = ops.arange(-12, 13, dtype=ms.float32).reshape(5, 5)
    print(x.norm(ord=1))