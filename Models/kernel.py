from os import error
import torch


class GaussianKernel:
    def __init__(self, param=None):
        ''' GaussianKernel
        (SignalPower)exp[-(0.5)*sum{(|x-x'|**2)/(LengthScales)}]
        param[0]: SignalPower
        param[1]: LengthScales

        SignalPower = torch.var(self.Y, 0)
        NoisePower = 0.125 * SignalPower
        lengthscales = torch.log((max(self.X) - min(self.X)) * 0.5)
        f_param = torch.cat([SignalPower, lengthscales])
        g_param = torch.cat([NoisePower, lengthscales])
        fkernel.__init__(param=f_param)
        gkernel.__init__(param=g_param)
        '''
        self.parameters = param
        if param is None:
            self.SignalPower = torch.log(torch.tensor(1.0))
            self.LengthScales = torch.log(torch.rand(1) * 0.1 + 0.5)
        elif len(param) == 1:
            self.SignalPower = torch.log(torch.tensor(1.0))
            self.LengthScales = torch.log(torch.rand(1) * 0.05 + param[0])
        else:
            self.SignalPower = 0.5 * torch.log(param[:-1])
            self.LengthScales = torch.log(param[-1])

    def K(self, x1, x2=None):
        if x2 is None:
            c = ((x1.t()[:, :, None] - x1.t()[:, None, :])
                 ** 2 / torch.exp(self.LengthScales)).sum(0)
        else:
            c = ((x1.t()[:, :, None] - x2.t()[:, None, :])
                 ** 2 / torch.exp(self.LengthScales)).sum(0)

        return (torch.exp(self.SignalPower) * torch.exp(-0.5 * c)).float()

    def param(self):
        if self.parameters is None:
            return [self.LengthScales]
        elif len(self.parameters) == 1:
            return [self.LengthScales]
        elif len(self.parameters) == 2:
            return [self.SignalPower, self.LengthScales]
        else:
            error

    def compute_grad(self, flag):
        if self.parameters is None:
            self.LengthScales.requires_grad = flag
        elif len(self.parameters) == 1:
            self.LengthScales.requires_grad = flag
        elif len(self.parameters) == 2:
            self.LengthScales.requires_grad = flag
            self.SignalPower.requires_grad = flag
        else:
            self.LengthScales.requires_grad = flag

    def param_set(self, X, Y):
        pass
