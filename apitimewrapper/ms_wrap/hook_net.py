from .wrap_Tensor import initialize_hook_tensor
from .wrap_function import initialize_hook_ops
from .wrap_nn import initialize_hook_nn


def start_hook_net(hook_inside=False, parall_execute=False):
    initialize_hook_tensor(hook_inside, parall_execute)
    initialize_hook_ops(hook_inside, parall_execute)
    initialize_hook_nn(parall_execute)
