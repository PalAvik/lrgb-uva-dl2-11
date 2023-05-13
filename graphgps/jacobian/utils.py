import torch
from torch.autograd.functional import *
from typing import Tuple, List
from torch_geometric.data import Data


def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _as_tuple(inp, arg_name=None, fn_name=None):
    # Ensures that inp is a tuple of Tensors
    # Returns whether or not the original inp was a tuple and the tupled version of the input
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)

    is_inp_tuple = True
    if not isinstance(inp, tuple):
        inp = (inp,)
        is_inp_tuple = False

    for i, el in enumerate(inp):
        if not isinstance(el, torch.Tensor):
            if is_inp_tuple:
                raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                " value at index {} has type {}.".format(arg_name, fn_name, i, type(el)))
            else:
                raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                " given {} has type {}.".format(arg_name, fn_name, arg_name, type(el)))

    return is_inp_tuple, inp


def _tuple_postprocess(res, to_unpack):
    # Unpacks a potentially nested tuple of Tensors
    # to_unpack should be a single boolean or a tuple of two booleans.
    # It is used to:
    # - invert _as_tuple when res should match the inp given to _as_tuple
    # - optionally remove nesting of two tuples created by multiple calls to _as_tuple
    if isinstance(to_unpack, tuple):
        assert len(to_unpack) == 2
        if not to_unpack[1]:
            res = tuple(el[0] for el in res)
        if not to_unpack[0]:
            res = res[0]
    else:
        if not to_unpack:
            res = res[0]
    return res


def _check_requires_grad(inputs, input_type, strict):
    # Used to make all the necessary checks to raise nice errors in strict mode.
    if not strict:
        return

    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            # This can only be reached for grad_inputs.
            raise RuntimeError("The output of the user-provided function is independent of input {}."
                               " This is not allowed in strict mode.".format(i))
        if not inp.requires_grad:
            if input_type == "hessian":
                raise RuntimeError("The hessian of the user-provided function with respect to input {}"
                                   " is independent of the input. This is not allowed in strict mode."
                                   " You should ensure that your function is thrice differentiable and that"
                                   " the hessian depends on the inputs.".format(i))
            elif input_type == "jacobian":
                raise RuntimeError("While computing the hessian, found that the jacobian of the user-provided"
                                   " function with respect to input {} is independent of the input. This is not"
                                   " allowed in strict mode. You should ensure that your function is twice"
                                   " differentiable and that the jacobian depends on the inputs (this would be"
                                   " violated by a linear function for example).".format(i))
            elif input_type == "grad_inputs":
                raise RuntimeError("The gradient with respect to input {} is independent of the inputs of the"
                                   " user-provided function. This is not allowed in strict mode.".format(i))
            else:
                raise RuntimeError("Output {} of the user-provided function does not require gradients."
                                   " The outputs must be computed in a differentiable manner from the input"
                                   " when running in strict mode.".format(i))


def _autograd_grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None, is_grads_batched=False):
    # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
    # This has the extra constraint that inputs has to be a tuple
    assert isinstance(outputs, tuple)
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    assert isinstance(grad_outputs, tuple)
    assert len(outputs) == len(grad_outputs)

    new_outputs: Tuple[torch.Tensor, ...] = tuple()
    new_grad_outputs: Tuple[torch.Tensor, ...] = tuple()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        # No differentiable output, we don't need to call the autograd engine
        return (None,) * len(inputs)
    else:
        return torch.autograd.grad(new_outputs, inputs, new_grad_outputs, allow_unused=True,
                                   create_graph=create_graph, retain_graph=retain_graph,
                                   is_grads_batched=is_grads_batched)

                
def _grad_preprocess(inputs, create_graph, need_graph):
    # Preprocess the inputs to make sure they require gradient
    # inputs is a tuple of Tensors to preprocess
    # create_graph specifies if the user wants gradients to flow back to the Tensors in inputs
    # need_graph specifies if we internally want gradients to flow back to the Tensors in res
    # Note that we *always* create a new Tensor object to be able to see the difference between
    # inputs given as arguments and the same Tensors automatically captured by the user function.
    # Check this issue for more details on how that can happen: https://github.com/pytorch/pytorch/issues/32576
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            # Create at least a new Tensor object in a differentiable way
            if not inp.is_sparse:
                # Use .view_as() to get a shallow copy
                res.append(inp.view_as(inp))
            else:
                # We cannot use view for sparse Tensors so we clone
                res.append(inp.clone())
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)


def _grad_postprocess(inputs, create_graph):
    # Postprocess the generated Tensors to avoid returning Tensors with history when the user did not
    # request it.
    if isinstance(inputs[0], torch.Tensor):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        else:
            return inputs
    else:
        return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)


def jacobian_graph(func, inputs, create_graph=False, strict=False, is_graphgym=False, uses_pe=False):
    r"""Function that computes the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the input.
    """

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if is_graphgym:
            if uses_pe:
                input_gdata = Data(
                        x=inputs[0],
                        edge_index=inputs[1].int(),
                        edge_attr=inputs[2],
                        EigVals=inputs[3],
                        EigVecs=inputs[4],
                    )
                outputs = (func(input_gdata)[0])
            else:
                input_gdata = Data(
                        x=inputs[0],
                        edge_index=inputs[1].int(),
                        edge_attr=inputs[2],
                    )
                outputs = (func(input_gdata)[0])
        else:
            outputs = func(*inputs)
        
        is_outputs_tuple, outputs = _as_tuple(outputs,
                                              "outputs of the user-provided function",
                                              "jacobian")
        _check_requires_grad(outputs, "outputs", strict=strict)

        jacobian: Tuple[torch.Tensor, ...] = tuple()

        for i, out in enumerate(outputs):

            # mypy complains that expression and variable have different types due to the empty list
            jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore[assignment]
            for j in range(out.nelement()):
                vj = _autograd_grad((out.reshape(-1)[j],), inputs,
                                    retain_graph=True, create_graph=create_graph)

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = ("The jacobian of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode when create_graph=True.".format(i))
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = ("Output {} of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode.".format(i, el_idx))
                            raise RuntimeError(msg)
                        jac_i_el.append(torch.zeros_like(inp_el))

            jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size()  # type: ignore[operator]
                         + inputs[el_idx].size()) for (el_idx, jac_i_el) in enumerate(jac_i)), )

        jacobian = _grad_postprocess(jacobian, create_graph)

        return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple))