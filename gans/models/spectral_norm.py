"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
import torch.nn.functional as F


def normalize(t, eps=1e-12):
    shape = t.size()
    t = t.view(-1)
    t = F.normalize(t, dim=0, eps=eps)
    t = t.view(shape)
    return t


class SpectralNorm(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                'Expected n_power_iterations to be positive, but '
                'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = weight

        if do_power_iteration:
            with torch.no_grad():
                # print(u, v)
                for _ in range(self.n_power_iterations):
                    if isinstance(module, torch.nn.Linear):
                        v = F.linear(u, weight_mat)
                        v = normalize(v, eps=self.eps)
                        u = F.linear(v, weight_mat.t())
                        u = normalize(u, eps=self.eps)
                    elif isinstance(module, torch.nn.Conv2d):
                        v = F.conv2d(
                            u, weight, None, module.stride, module.padding,
                            module.dilation, module.groups)
                        v = normalize(v, eps=self.eps)
                        u = F.conv_transpose2d(
                            v, weight, None, module.stride, module.padding, 0,
                            module.groups, module.dilation)
                        u = normalize(u, eps=self.eps)
                    else:
                        raise ValueError("Module %s is not supported" %
                                         module.__class__.__name__)

                    # v = normalize(torch.mv(weight_mat.t(), u),
                    #               dim=0, eps=self.eps, out=v)
                    # u = normalize(torch.mv(weight_mat, v),
                    #               dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    # u = u.clone(memory_format=torch.contiguous_format)
                    # v = v.clone(memory_format=torch.contiguous_format)
                    u = u.clone()
                    v = v.clone()
        if isinstance(module, torch.nn.Linear):
            forward_u = F.linear(u, weight_mat)
        elif isinstance(module, torch.nn.Conv2d):
            forward_u = F.conv2d(
                u, weight, None, module.stride, module.padding,
                module.dilation, module.groups)
        else:
            raise ValueError(
                "Module %s is not supported" % module.__class__.__name__)
        sigma = torch.dot(v.view(-1), forward_u.view(-1))
        # print(sigma)
        k = 1
        weight = weight / sigma * k
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(
            self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(
            module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(),
                               weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name))

        dim = [1] + list(dim)
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            # weight_mat = fn.reshape_weight_to_matrix(weight)

            # h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(dim).normal_(0, 1), eps=fn.eps)
            if isinstance(module, torch.nn.Linear):
                v = F.linear(u, weight)
            elif isinstance(module, torch.nn.Conv2d):
                v = F.conv2d(
                    u, weight, None, module.stride, module.padding,
                    module.dilation, module.groups)
            else:
                raise ValueError(
                    "Module %s is not supported" % module.__class__.__name__)
            v = normalize(weight.new_empty(v.size()).normal_(0, 1), eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # plain attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(
            SpectralNormLoadStateDictPreHook(fn))
        return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get(
            'spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if version is None and all(weight_key + s in state_dict
                                       for s in ('_orig', '_u', '_v')) and \
                    weight_key not in state_dict:
                # Detect if it is the updated state dict and just missing
                # metadata. This could happen if the users are crafting a state
                # dict themselves, so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ('_orig', '', '_u'):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + '_orig']
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + '_v'] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError(
                "Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


def spectral_norm(module, dim, name='weight', n_power_iterations=1, eps=1e-12):
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("spectral_norm of '{}' not found in {}".format(
            name, module))

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormStateDictHook) and \
                hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, SpectralNormLoadStateDictPreHook) and \
                hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break

    return module
