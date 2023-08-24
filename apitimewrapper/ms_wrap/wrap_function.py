import os
import yaml
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.api import _pynative_executor
import mindspore.ops.function as F
import time
from . import global_param
from .my_print import print

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('ops')

OpsFunc = {}
for f in dir(ops):
    OpsFunc[f] = getattr(ops, f)


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(ms.ops)
    return set(WrapFunctionalOps) & set(_all_functional_ops)


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(nn.Cell):
    def __init__(self, op_name, hook_inside=False, parall_execute=False):
        super(FunctionalOPTemplate, self).__init__()
        self.op_name_ = op_name
        self.changed_status = hook_inside
        self.parall_execute = parall_execute
        if not global_param.g_stop_hook:
            global_param.g_stop_hook = True
            self.changed_status = True

    def construct(self, *args, **kwargs):
        if self.changed_status:
            start_time = time.perf_counter()
            out = OpsFunc[self.op_name_](*args, **kwargs)
            if not self.parall_execute:
                _pynative_executor.sync()
            end_time = time.perf_counter()
            print(f"ops.{self.op_name_} cost_time:{end_time - start_time}")
            self.changed_status = False
            global_param.g_stop_hook = False
        else:
            out = OpsFunc[self.op_name_](*args, **kwargs)
        return out


def wrap_functional_op(op_name, hook_inside=False, parall_execute=False):
    def functional_op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name, hook_inside, parall_execute)(*args, **kwargs)

    return functional_op_template


def wrap_functional_ops_and_bind(hook_inside=False, parall_execute=False):
    for op_name in get_functional_ops():
        setattr(HOOKFunctionalOP, "wrap_" + op_name, wrap_functional_op(op_name, hook_inside, parall_execute))


def initialize_hook_ops(hook_inside=False, parall_execute=False):
    wrap_functional_ops_and_bind(hook_inside, parall_execute)
    for attr_name in dir(HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(ops, attr_name[5:], getattr(HOOKFunctionalOP, attr_name))


if __name__ == '__main__':
    initialize_hook_ops(hook_inside=False, parall_execute=False)
    x = ops.arange(-12, 13, dtype=ms.float32)
    print(ops.norm(x))