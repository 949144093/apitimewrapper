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
pip install apitimewrapper-0.0.4-py3-none-any.whl
```

#### 使用说明
1. pip install apitimewrapper-0.0.4-py3-none-any.whl
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
