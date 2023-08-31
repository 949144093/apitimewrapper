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
from .my_print import print

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('functional')

for f in dir(torch.nn.functional):
    locals().update({f: getattr(torch.nn.functional, f)})


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(torch.nn.functional)
    return set(WrapFunctionalOps) & set(_all_functional_ops)


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(torch.nn.Module):
    def __init__(self, op_name, hook_inside=False, parall_execute=False):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Functional_" + str(op_name) + "_"
        super(FunctionalOPTemplate, self).__init__()
        self.changed_status = hook_inside
        self.parall_execute = parall_execute
        if not global_param.g_stop_hook:
            global_param.g_stop_hook = True
            self.changed_status = True

    def forward(self, *args, **kwargs):
        if self.changed_status:
            try:
                start_time = time.perf_counter()
                out = eval(self.op_name_)(*args, **kwargs)
                if not self.parall_execute:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                print(f"torch.nn.function.{self.op_name_} cost_time:{end_time - start_time}")
            except Exception as e:
                raise e
            finally:
                self.changed_status = False
                global_param.g_stop_hook = False
        else:
            out = eval(self.op_name_)(*args, **kwargs)
        return out


def wrap_functional_op(op_name, hook_inside=False, parall_execute=False):
    def functional_op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name, hook_inside, parall_execute)(*args, **kwargs)

    return functional_op_template


def wrap_functional_ops_and_bind(hook_inside=False, parall_execute=False):
    _functional_ops = get_functional_ops()
    for op_name in _functional_ops:
        setattr(HOOKFunctionalOP, "wrap_" + op_name, wrap_functional_op(op_name, hook_inside, parall_execute))


def initialize_hook_function(hook_inside=False, parall_execute=False):
    wrap_functional_ops_and_bind(hook_inside, parall_execute)
    for attr_name in dir(HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(torch.nn.functional, attr_name[5:], getattr(HOOKFunctionalOP, attr_name))