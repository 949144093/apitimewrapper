# apitimewrapper

#### 介绍
使用wrap机制对mindspore及pytorch的所有接口进行自动打点，统计API执行性能

#### 软件架构

```
.
├── apitimewrapper
│   ├── analysis.py
│   ├── __init__.py
│   ├── ms_wrap
│   │   ├── global_param.py
│   │   ├── hook_net.py
│   │   ├── __init__.py
│   │   ├── my_print.py
│   │   ├── performance_cmp_new.py
│   │   ├── support_wrap_ops.yaml
│   │   ├── tracker.py
│   │   ├── wrap_function.py
│   │   ├── wrap_nn.py
│   │   └── wrap_Tensor.py
│   └── torch_wrap
│       ├── global_param.py
│       ├── hook_net.py
│       ├── __init__.py
│       ├── my_print.py
│       ├── support_wrap_ops.yaml
│       ├── wrap_functional.py
│       ├── wrap_module.py
│       ├── wrap_tensor.py
│       └── wrap_torch.py
├── setup.py
└── setup.sh

```



#### 安装教程

```
bash setup.sh
cd dist
pip install apitimewrapper-0.0.5-py3-none-any.whl
```

#### 使用说明
1. pip install apitimewrapper-0.0.5-py3-none-any.whl
2. 修改我们的网络执行入口文件，若要执行训练，则修改train.py, 若要执行推理，则修改eval.py。  
以如下dino网络为例：  
首先在文件注释step1的位置增添导包，分别导出start_hook_net和print方法，其中start_hook_net方法用于对我们整网的所有api(nn, ops, tensor)进行wrap，在其执行前后进行自动打点计时，print则重载了原生的内建print方法，增添了打屏并写入日志的功能。  
其次在文件注释step2的位置启动wrap功能，此操作务必要放在网络执行前，保证在执行网络前所有的api已被进行全量替换，其中hook_inside参数代表我们进行api打点计时时是否要对使用api内部封装逻辑调用的api进行打点，例如网络内部使用了ops.norm接口进行计算，我们在计算ops.norm时间后是否要对norm内部实现调用的sqrt，square等api进行计时，默认为Fasle，表示只对网络内部使用的一级api进行打点计时。  
```python
# step1 增添导包
################################
from apitimewrapper import start_hook_net, print
################################

if __name__ == '__main__':
    # step2 增添启动代码
    ################################
    hook_inside = False
    start_hook_net(hook_inside)
    ################################


    # create dataset
    ...
    # load pretrained model, only load backbone
    ...
    # create model with loss scale
    ...
    # training loop
    ...
```

#### 0.0.4版本功能新增说明
##### 功能新增说明
性能分析工具0.0.4版本相对于0.0.3版本新增参数parall_execute，参数默认为False，功能与0.0.3版本兼容，若单算子接口串行测试性能不达标，则可以开启异步进行测试。
将测试脚本中的
```python
start_hook_net(hook_inside)
start_hook_torch_net(hook_inside)
```
修改为:  
```python
start_hook_net(hook_inside, parall_execute=True)
start_hook_torch_net(hook_inside, parall_execute=True)
```

此外，在测试脚本中for循环结束后增添异步等待的代码，具体如下：  
将测试脚本中的
```python
start_analysis()
for _ in range(100):
    ops.reshape(ms_input_a, (4000, 500))

for _ in range(100):
    torch.reshape(torch_input_a, (4000, 500))
end_analysis()

```
修改为:  
```python
start_analysis()
for _ in range(100):
    ops.reshape(ms_input_a, (4000, 500))
_pynative_executor.sync()
for _ in range(100):
    torch.reshape(torch_input_a, (4000, 500))
torch.cuda.synchronize()
end_analysis()
```

##### 完整测试用例示例  
```python
import os

import torch
import mindspore
import mindspore as ms
from mindspore import ops, nn, Tensor

import numpy as np
from apitimewrapper import start_hook_net, start_hook_torch_net, print, start_analysis, end_analysis
from mindspore.common.api import _pynative_executor

# 需自定义测试的部分
#########################################
# np格式的输入
input_a = np.random.randn(1000, 2000)
input_b = np.random.randn(1000, 2000)

# 转为ms的tensor
ms_input_a = Tensor(input_a, mindspore.float32)
ms_input_b = Tensor(input_b, mindspore.float32)

# 转为torch的tensor
torch_input_a = torch.tensor(input_a, dtype=torch.float32)
torch_input_b = torch.tensor(input_b, dtype=torch.float32)

# 手动将torch的Tensor拷贝到对应设备上，卡号需手动修改
device_id = int(os.environ.get('DEVICE_ID', 0))
torch_input_a.to(device=torch.device(f'cuda:{device_id}'))
torch_input_b.to(device=torch.device(f'cuda:{device_id}'))

hook_inside = False
start_hook_net(hook_inside, parall_execute=True)
start_hook_torch_net(hook_inside, parall_execute=True)

"""
这段代码的作用是为了先启动框架，排除框架首次启动时长以及host侧到device侧tensor拷贝时长的耗时影响。
"""
ms_input_a.reshape((4000, 500))
torch_input_a.reshape((4000, 500))
ms_input_a.reshape((4000, 500))
torch_input_a.reshape((4000, 500))
print("------排除框架启动执行耗时------")

# 在0.0.3版本，我们可以通过在执行代码前后增添start_analysis()和end_analysis()的方式，自动对执行部分代码进行性能分析。
start_analysis()
for _ in range(100):
    # ms_input_a.reshape((4000, 500))
    ops.reshape(ms_input_a, (4000, 500))
_pynative_executor.sync()
for _ in range(100):
    # torch_input_a.reshape((4000, 500))
    torch.reshape(torch_input_a, (4000, 500))
torch.cuda.synchronize()
end_analysis()

```

#### 0.0.5版本更新日志
1. 新增全局变量控制运行时是否打屏（默认写入日志）
```python
from apitimewrapper.ms_wrap import global_param as ms_gp
from apitimewrapper.torch_wrap import global_param as torch_gp
ms_gp.g_stop_ms_print = True
torch_gp.g_stop_torch_print = True
```
2. 新增异常处理逻辑，修复在测试脚本中使用异常处理时导致后续api性能数据消失的bug
3. 增添torch遗漏接口torch.meshgrid, torch.Tensor.split

#### 0.0.6版本更新日志
1. 修复__add__ __mul__等魔法方法无法统计性能的问题，针对原生使用的+ - * /等提供支持

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
