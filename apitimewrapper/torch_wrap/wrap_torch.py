#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import time

import torch
import yaml
from . import global_param
# import global_param
from .my_print import print

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapTorchOps = yaml.safe_load(f).get('torch')


def get_torch_ops():
    global WrapTorchOps
    _torch_ops = dir(torch._C._VariableFunctionsClass)
    return set(WrapTorchOps) & set(_torch_ops)


class HOOKTorchOP(object):
    pass


class TorchOPTemplate(torch.nn.Module):

    def __init__(self, op_name, hook_inside=False, parall_execute=False):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Torch_" + str(op_name) + "_"
        super().__init__()
        self.changed_status = hook_inside
        self.parall_execute = parall_execute
        if not global_param.g_stop_hook:
            global_param.g_stop_hook = True
            self.changed_status = True

    def input_param_need_adapt(self):
        special_op_list = ["broadcast_tensors"]
        for item in special_op_list:
            if item in self.op_name_:
                return True
        return False

    def einsum_adapt(self, *args):
        if len(args) < 2:
            raise ValueError('einsum(): must specify the equation string and at least one operand, '
                             'or at least one operand and its subscripts list')
        equation = None
        operands = None
        if isinstance(args[0], torch.Tensor):
            def parse_subscript(n: int) -> str:
                if n == Ellipsis:
                    return '...'
                if n >= 0 and n < 26:
                    return chr(ord('A') + n)
                if n >= 26 and n < 52:
                    return chr(ord('a') + n - 26)
                raise ValueError('einsum(): subscript in subscript list is not within the valid range [0, 52]')

            equation = ','.join(''.join(parse_subscript(s) for s in l) for l in args[1::2])

            if len(args) % 2 == 1:
                equation += '->' + ''.join(parse_subscript(s) for s in args[-1])
                operands = args[:-1:2]
            else:
                operands = args[::2]
        else:
            equation = args[0]
            operands = args[1:]

        if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            _operands = operands[0]
            return self.einsum_adapt(equation, *_operands)
        return equation, operands

    def forward(self, *args, **kwargs):
        if self.changed_status:
            try:
                start_time = time.perf_counter()
                if self.input_param_need_adapt():
                    out = getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(args, **kwargs)
                else:
                    if self.op_name_ == 'einsum':
                        args = self.einsum_adapt(*args)
                    out = getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)
                if not self.parall_execute:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                print(f"torch.{self.op_name_} cost_time:{end_time - start_time}")
            except Exception as e:
                raise e
            finally:
                self.changed_status = False
                global_param.g_stop_hook = False
        else:
            if self.input_param_need_adapt():
                out = getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(args, **kwargs)
            else:
                if self.op_name_ == 'einsum':
                    args = self.einsum_adapt(*args)
                out = getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)
        return out


def wrap_torch_op(op_name, hook_inside=False, parall_execute=False):
    def torch_op_template(*args, **kwargs):
        return TorchOPTemplate(op_name, hook_inside, parall_execute)(*args, **kwargs)

    return torch_op_template


def wrap_torch_ops_and_bind(hook_inside=False, parall_execute=False):
    _torch_ops = get_torch_ops()
    for op_name in _torch_ops:
        setattr(HOOKTorchOP, "wrap_" + op_name, wrap_torch_op(op_name, hook_inside, parall_execute))


def initialize_hook_torch(hook_inside=False, parall_execute=False):
    wrap_torch_ops_and_bind(hook_inside, parall_execute)
    for attr_name in dir(HOOKTorchOP):
        if attr_name.startswith("wrap_"):
            setattr(torch, attr_name[5:], getattr(HOOKTorchOP, attr_name))
