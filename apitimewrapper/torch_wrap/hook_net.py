from .wrap_functional import initialize_hook_function
from .wrap_module import initialize_hook_module
from .wrap_tensor import initialize_hook_tensor
from .wrap_torch import initialize_hook_torch


def start_hook_torch_net(hook_inside=False, parall_execute=False):
    initialize_hook_function(hook_inside, parall_execute)
    initialize_hook_tensor(hook_inside, parall_execute)
    initialize_hook_torch(hook_inside, parall_execute)
    initialize_hook_module(parall_execute)