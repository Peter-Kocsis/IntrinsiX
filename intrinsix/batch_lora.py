"""Based on https://github.com/cloneofsimo/lora/blob/d84074b3e3496f1cfa8a3f49b8b9972ef463b483/lora_diffusion/lora.py#L255"""
from typing import Set, Dict, Optional, Type, List

import torch
from torch import nn
from torch.nn import Sequential


class LoraLinear(nn.Module):
    def __init__(
        self, in_features, out_features, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (
            self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )


class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        # self.linear = nn.Linear(in_features, out_features, bias)
        self.linear = None
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (
            self.linear(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)



class Batchwise(Sequential):
    """
    Module to run different modules over a batch
    """
    def forward(self, input):
        # TODO: implement fused version if needed
        assert input.shape[0] == len(self._modules), f"Batch size must match the number of modules {input.shape[0]} != {len(self._modules)}"
        input = torch.cat([module(x.unsqueeze(0)) for x, module in zip(input, self)], dim=0)
        return input


def inject_trainable_batched_lora(
    model: nn.Module,
    target_modules: Set[str],
    lora_configs: list,
    verbose: bool = False
):
    """
    Inject loras into model to be evaluated within a batch
    """
    # Find the target modules
    for module_path, _module, name, _child_module in find_modules(model, target_module_names=target_modules, search_class=[nn.Linear]):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection: injecting into", module_path)

        # Create the lora module
        loras = [LoraInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
                r=config['r'],
                dropout_p=config['dropout_p'],
                scale=config['scale']
            ) if config is not None else _child_module for config in lora_configs]

        # Move the original layer inside
        for lora in loras:
            if not isinstance(lora, LoraInjectedLinear):
                continue
            
            # lora.linear.weight = weight
            # if bias is not None:
            #     lora.linear.bias = bias
            lora.linear = _child_module
            
            # Make the new modules trainable
            lora.lora_up.weight.requires_grad = True
            lora.lora_down.weight.requires_grad = True
        
        # Create the batched module
        lora_module = Batchwise(*loras)

        # Replace the module
        lora_module.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = lora_module

    return model


def find_modules(
    model,
    ancestor_class: Optional[Set[str]] = ["Attention"],
    target_module_names: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedLinear,
    ],
):
    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            (name, module)
            for name, module in model.named_modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [(name, module) for module in model.named_modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor_name, ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                if target_module_names is not None and all(target not in fullname for target in target_module_names):
                    continue
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it

                module_path = ".".join([ancestor_name, fullname])
                yield module_path, parent, name, module


def extract_loras(model):
    loras = {}

    for module_path, _m, _n, _child_module in find_modules(
        model,
        search_class=[LoraInjectedLinear],
    ):
        lora_state_dict = _child_module.state_dict()
        keys_to_delete = [key for key in lora_state_dict.keys() if "linear" in key]
        for keys in keys_to_delete:
            del lora_state_dict[keys]
        lora_state_dict = {module_path + "." + k: v for k, v in lora_state_dict.items()}

        loras.update(lora_state_dict)

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras


def save_lora_weights(
    model,
    path="./lora.pt"
):
    loras_state_dict = extract_loras(model)
    torch.save(loras_state_dict, path)
