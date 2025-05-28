import torch
from torch.autograd.grad_mode import _unsafe_preserve_version_counter


def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def zeroed_versions(model: torch.nn.Module):
    for param in model.parameters():
        param._version = 0


class preserve_version_context:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.saved_versions = {}

    def __enter__(self):
        for param in self.model.parameters():
            if param.requires_grad:
                self.saved_versions[param] = param._version

    def __exit__(self, exc_type, exc_value, traceback):
        for param, saved_version in self.saved_versions.items():
            with _unsafe_preserve_version_counter(param):
                torch._C._autograd._unsafe_set_version_counter(param, saved_version)


class untouched:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __enter__(self):
        freeze(self.model)

    def __exit__(self, exc_type, exc_value, traceback):
        unfreeze(self.model)


def get_versions(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if param._version != 0:
            print(f"Pas à la version de base {name}, {param._version}")
        else:
            print(f"À la version de base {name}, {param._version}")


def weight_mse(teacher: torch.nn.Module, student: torch.nn.Module):
    mse = 0.0
    for p_teacher, p_student in zip(teacher.parameters(), student.parameters()):
        mse += torch.mean((p_teacher - p_student) ** 2)
    return mse.detach()
