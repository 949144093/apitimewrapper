analysis.py 核心逻辑

```
from .ms_wrap import print
import builtins


def start_analysis():
    print("start_analysis")


def end_analysis():
    print("end_analysis")
    line_index = {}
    with open("performance.log", 'r') as f:
        all_data = f.readlines()
        for index, line in enumerate(all_data):
            if "start_analysis" in line:
                line_index["start_analysis"] = index
            if "end_analysis" in line:
                line_index["end_analysis"] = index
        analysis_time_data = all_data[line_index["start_analysis"]+1:line_index["end_analysis"]]
        result_dc = {}
        for data in analysis_time_data:
            op_name = data.split(" cost_time:")[0].split("INFO: ")[1]
            cost_time = eval(data.split(" cost_time:")[1].strip())
            if op_name not in result_dc:
                # 总耗时，执行此时，最低耗时，最高耗时
                result_dc[op_name] = [cost_time, 1, cost_time, cost_time]
            else:
                result_dc[op_name][0] += cost_time
                result_dc[op_name][1] += 1
                if cost_time < result_dc[op_name][2]:
                    result_dc[op_name][2] = cost_time
                if cost_time > result_dc[op_name][3]:
                    result_dc[op_name][3] = cost_time
        for op_name, op_info in result_dc.items():
            builtins.print(f"{op_name}共执行了{op_info[1]}次,总耗时{op_info[0]}秒,平均耗时{op_info[0]/op_info[1]}秒,最低耗时{op_info[2]},最高耗时{op_info[3]}.")
```


ms_wrap:
my_print:

```
import builtins
import logging


def print(*args, **kwargs):
    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        filename='performance.log',
                        filemode='w',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s'
                        # 日志格式
                        )
    logging.info(*args, **kwargs)
    return builtins.print(*args, **kwargs)
```

performance_cmp_new.py

```
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

```

wrap_function.py

```
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

```

wrap_nn.py

```
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
            start_time = time.perf_counter()
            result = original_call(self, *args, **kwargs)
            if not parall_execute:
                _pynative_executor.sync()
            end_time = time.perf_counter()
            print(f"nn.{self.cls_name} cost_time:{end_time - start_time}")

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

```
wrap_Tensor.py

```
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

```


