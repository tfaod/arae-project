import torch
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


'''
input_size: list of tuples/lists of input sizes. list shapes correspond to list shape where 0th element is the value, pytorch  
            tensors correspond to tuple shape
            pass in 1-tuples as (x,) as opposed to (x) to avoid bugs
'''

def summary(model,input_size,batch_size=64,cuda=True, outf="", dtype=torch.FloatTensor):

    '''
    HELPER FUNCTIONS
    '''

    def getSize(m,batch_size=batch_size):
        if type(m) is torch.Tensor or type(m) is Variable:
            return [batch_size]+list(m.shape)
        elif type(m) is nn.utils.rnn.PackedSequence:
            return m.data.shape
        elif type(m) is list or type(m) is tuple:
            return [getSize(l) for l in list(m) ]
        else:
            try:
                return m.data.shape
            except:
                raise TypeError("Trying to get size of output of type {}".format(type(m)))
                return -1

    def buildList(shape):
        if len(shape) == 1: return shape[0]
        out = 1
        for i in shape[1:]:
            out = [out]*i
        return out


    # handle inputs to python function
    def buildInputs(size,dtype,batch_size=batch_size, cuda=cuda):
        # check if there are multiple inputs to the network
        if isinstance(size, (list)):
            return buildList(size)
        else:
            assert(type(size) is tuple)
            if cuda:
                return Variable(torch.rand([batch_size] + list(size)).cuda()).type(dtype)
            else:
                return Variable(torch.rand([batch_size] + list(size))).type(dtype)

    
    def register_hook(module):
        
        
        
        def hook(module, input, output):
            # get module name, index
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            # get input output shape
            m_key = "{}-{}".format(class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(getSize(input[0]))
            summary[m_key]['output_shape'] = list(getSize(output[0]))

            # count parameters
            params = 0            
            if hasattr(module, 'weight'):
                params += np.prod(list(module.weight.size()))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias'):
                params += np.prod(summary[m_key]['output_shape'])                
            summary[m_key]['param_ct'] = params

        # if module is not sequential or module list, run forward hook
        if not isinstance(module, nn.Sequential) and \
             not isinstance(module, nn.ModuleList) and \
             not (module == model):
            hooks.append(module.register_forward_hook(hook))
            
    '''
    torch_summary CALLS 
    '''
    
    input_size = tuple([input_size,1]) if type(input_size) is int else input_size
    input_size = list([input_size]) if type(input_size) is tuple else input_size
    
    assert( isinstance(input_size, (list)))
    x = tuple([buildInputs(s,dtype) for s in input_size])
        
    # create properties
    summary = OrderedDict()
    hooks = []
    
    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)
    
    # remove these hooks
    for h in hooks:
        h.remove()

    if outf is "":
        print('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        print(line_new)
        print('================================================================')
        total_params = 0
        trainable_params = 0
        for layer in summary:
            ## input_shape, output_shape, trainable, param_ct
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['param_ct'])
            total_params += summary[layer]['param_ct']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['param_ct']
            print(line_new)
        print('================================================================')
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str(total_params - trainable_params))
        print('----------------------------------------------------------------')
    else:
        f = open(outf,"a")
        f.write('----------------------------------------------------------------')
        f.write('\n')
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        f.write('\n')
        f.write(line_new)
        f.write('\n')
        f.write('================================================================')
        f.write('\n')
        total_params = 0
        trainable_params = 0
        for layer in summary:
            ## input_shape, output_shape, trainable, param_ct
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['param_ct'])

            total_params += summary[layer]['param_ct']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['param_ct']
            f.write(line_new)
            f.write('\n')
        f.write('================================================================')
        f.write('\n')
        f.write('Total params: ' + str(total_params))
        f.write('\n')
        f.write('Trainable params: ' + str(trainable_params))
        f.write('\n')
        f.write('Non-trainable params: ' + str(total_params - trainable_params))
        f.write('\n')
        f.write('----------------------------------------------------------------')
        f.write('\n\n\n')

    return summary