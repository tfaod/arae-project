import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
import pdb

def summary(model, input_size, batch_size=64, device="cuda"):

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "{}-{}".format(class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            m_in = input[0]
                summary[m_key]["input_shape"] = getSize(input[0])

            def getSize(m,batch_size=64):
                if type(m) is torch.Tensor:
                    return [batch_size]+list(m.shape)
                elif type(m) is nn.utils.rnn.PackedSequence:
                    return m.data.shape
                elif type(m) is list:
                    return [getSize(l) for l in m ]
                elif type(m) is tuple:
                    return tuple([getSize(l) for l in list(m)])
                else:
                    raise TypeError("Trying to get size of output of type {}".format(type(m)))
                return None

            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(getSize(o))[1:] for o in output]
                
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
                
        if(not isinstance(module, nn.Sequential)
           and not isinstance(module, nn.ModuleList)
           and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))
                   
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    print("input_size has type {}".format(type(input_size)))

    if isinstance(input_size, tuple):
        input_size = list([input_size])
    if type(input_size) is int:
        input_size = list([tuple([input_size])])
    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
            print(line_new)

    # assume 4 bytes/number (float on cuda).
    _size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    _output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    _params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    _size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
