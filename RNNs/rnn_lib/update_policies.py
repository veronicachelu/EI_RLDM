import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from rnn_lib import utils

class EiDense_UpdatePolicy2():
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g
    to be positive.

    It assumes the layer has the parameters of an EiDenseWithShunt layer.

    This should furthest right when composed with mixins for correct MRO.
    '''
    def __init__(self) -> None:
        print("""EiDense_UpdatePolicy is Deprecated, and doesn't contain
        the correction either please use 
        (BaseUpdatePolicy, cSGD_Mixin, SGD, EiDense_clamp_mixin) 
        for DANN mlps""")

    def update(self, layer, **kwargs):
        """
        Args:
            lr : learning rate
        """
        # lr = kwargs['lr']
        with torch.no_grad():
            # if hasattr(layer, 'g'):
            #     layer.g -= layer.g.grad *lr
            # if layer.b.requires_grad:
            #     layer.b -= layer.b.grad *lr
            # layer.Wex -= layer.Wex.grad * lr
            # layer.Wei -= layer.Wei.grad * lr
            # layer.Wix -= layer.Wix.grad * lr
            # if hasattr(layer, 'alpha'):
            #     layer.alpha -= layer.alpha.grad *lr

            layer.Wix.data = torch.clamp(layer.Wix, min=0)
            layer.Wex.data = torch.clamp(layer.Wex, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            if hasattr(layer, 'g'):
                layer.g.data   = torch.clamp(layer.g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()


class EiRNN_UpdatePolicy2():
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g
    to be positive.

    It assumes the layer has the parameters of an EiDenseWithShunt layer.

    This should furthest right when composed with mixins for correct MRO.
    '''
    def __init__(self) -> None:
        print("To refactor!, use (SGD, EiRNN_clamp_mixin) instead")

    def update(self, layer, **args):
        """
        Args:
            lr : learning rate
        """
        # lr = args['lr']
        with torch.no_grad():
            # layer.b -= layer.b.grad *lr
            # layer.Uex -= layer.Uex.grad * lr
            # layer.Uei -= layer.Uei.grad * lr
            # layer.Uix -= layer.Uix.grad * lr
            # layer.Wex -= layer.Wex.grad * lr
            # layer.Wei -= layer.Wei.grad * lr
            # layer.Wix -= layer.Wix.grad * lr
            # if hasattr(layer, 'U_alpha'):
            #     layer.U_alpha -= layer.U_alpha.grad *lr
            # if hasattr(layer, 'W_alpha'):
            #     layer.W_alpha -= layer.W_alpha.grad *lr
            # if hasattr(layer, 'U_g'):
            #     layer.U_g -= layer.U_g.grad *lr
            # if hasattr(layer, 'W_g'):
            #     layer.W_g -= layer.W_g.grad *lr

            layer.Wix.data = torch.clamp(layer.Wix, min=0)
            layer.Wex.data = torch.clamp(layer.Wex, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            layer.Uix.data = torch.clamp(layer.Uix, min=0)
            layer.Uex.data = torch.clamp(layer.Uex, min=0)
            layer.Uei.data = torch.clamp(layer.Uei, min=0)
            if hasattr(layer, 'U_g'):
                layer.U_g.data   = torch.clamp(layer.U_g, min=0)
            if hasattr(layer, 'W_g'):
                layer.W_g.data   = torch.clamp(layer.W_g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()

#--------------- Song SGD -----------------

class ColumnEiDenseSGD:
    def __init__(self, max=None, ablate_ii=False):
        self.max = max
        self.ablate_ii = ablate_ii
    
    @torch.no_grad()
    def update(self, layer, **args):
        # b_norm = layer.b.grad.norm(2)
        # W_pos_norm = layer.W_pos.grad.norm(2)
        # # gradient clipping
        # if self.max is not None:
        #     if b_norm > self.max:
        #         layer.b.grad *= (self.max / b_norm)
        #     if W_pos_norm > self.max:
        #         layer.W_pos.grad *= (self.max / W_pos_norm)

        # lr = args['lr']
        # layer.b     -= layer.b.grad *lr
        # layer.W_pos -= layer.W_pos.grad * lr

        if layer.clamp:
            layer.W_pos.data = torch.clamp(layer.W_pos, min=0)
        if self.ablate_ii:
            layer.W_pos.data[-layer.ni:,-layer.ni:] = 0

class ColumnEiSGD_Clip:
    def __init__(self, max=None, ablate_ii=False):
        self.max = max
        self.ablate_ii = ablate_ii

    @torch.no_grad()
    def update(self, layer, **args):
        # b_norm = layer.b.grad.norm(2)
        # W_pos_norm = layer.W_pos.grad.norm(2)
        # U_pos_norm = layer.U_pos.grad.norm(2)
        # # gradient clipping
        # if self.max is not None:
        #     if b_norm > self.max:
        #         layer.b.grad *= (self.max / b_norm)
        #     if W_pos_norm > self.max:
        #         layer.W_pos.grad *= (self.max / W_pos_norm)
        #     if U_pos_norm > self.max:
        #         layer.U_pos.grad *= (self.max / U_pos_norm)
        
        # lr = args['lr']
        # layer.b     -= layer.b.grad *lr
        # layer.W_pos -= layer.W_pos.grad * lr
        # layer.U_pos -= layer.U_pos.grad * lr

        if layer.clamp:
            layer.W_pos.data = torch.clamp(layer.W_pos, min=0)
            layer.U_pos.data = torch.clamp(layer.U_pos, min=0)
        if self.ablate_ii:
            layer.W_pos.data[-layer.ni:,-layer.ni:] = 0

class EiRNN_cSGD_UpdatePolicy2(EiRNN_UpdatePolicy2):
    def __init__(self, max_grad_norm=None):
        super(EiRNN_cSGD_UpdatePolicy2, self).__init__()
        self.max_grad_norm=max_grad_norm


class DalesANN_cSGD_UpdatePolicy2(EiDense_UpdatePolicy2):
    def __init__(self, max_grad_norm=None):
        super(DalesANN_cSGD_UpdatePolicy2, self).__init__()
        self.max_grad_norm=max_grad_norm




if __name__ == "__main__":
    # run in the shell 'export PYTHONPATH=$PYTHONPATH:~/dann-rnns'
    pass
