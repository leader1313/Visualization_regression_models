import torch
import numpy as np

'''
To do
1) LBFGS parameter surch

'''


class VHGPR:
    def __init__(self, X, Y, fkernel, gkernel):
        '''
        VHGPR - MV bound for heteroscedastic GP regression
            Input:
                - X[N X Q]: Training input data. One vector per row.
                - y[N X D]: Training output value. One scalar per row.
                - N: Size of dataset
                - D: Dimension of Y
                - fkernel: Covariance function for the GP f (signal).
                - gkernel: Covariance function for the GP g (noise).
                - Hyperparameters = {Lamda[D X N], hyperf[1], hyperg[1], mu0[D]}
        The algorithm in this file is based on the following paper:
        M. Lazaro Gredilla and M. Titsias,
        "Variational Heteroscedastic Gaussian Process Regression"
        Published in ICML 2011
        Copyright (c) 2021 by Hanbit Oh
        '''
        self.X = X
        self.Y = Y
        self.N, self.D = self.Y.shape
    # Hyperparameters Initialization
        # lengthscales = torch.log((max(self.X) - min(self.X)) * 0.5)
        # f_param = torch.cat([lengthscales])
        # g_param = torch.cat([lengthscales])
        # fkernel.__init__(param=f_param)
        # gkernel.__init__(param=g_param)
        self.fkern = fkernel  # f: policy function
        self.gkern = gkernel  # g: noise function
        SignalPower = torch.var(self.Y, 0)
        NoisePower = 0.125 * SignalPower
        # kernel can be initialized by
        # VHGPR.__init__()
        # Gaussian approximation parameters
        self.Lambda = torch.log(torch.tensor(0.5)) * torch.ones(self.D, self.N)
        self.mu0 = torch.log(NoisePower)  # [D]

    def Precomputation(self):
        Kf = self.fkern.K(self.X)
        Kg = self.gkern.K(self.X)
        Lambda = torch.exp(self.Lambda)         # [D X N]
        mu0 = torch.exp(self.mu0)               # [D]

        sLambda = torch.sqrt(Lambda).unsqueeze(2)  # [D X N X 1]
        Kgscaled = Kg.mul(sLambda.bmm(
            sLambda.transpose(1, 2)))  # [D X N X N]
        cinvB, LU = torch.solve(torch.eye(self.N), torch.cholesky(
            torch.eye(self.N) + Kgscaled, upper=False))  # [D X N X N]
        A = torch.ones(self.D, self.N, 1).bmm(sLambda.transpose(1, 2))
        cinvBs = cinvB.mul(A)
        beta = (Lambda - 0.5).t()  # [N X D]
        repeated_mu0 = mu0[:, None].repeat_interleave(
            self.N, dim=1)  # [D X N]
        mu = Kg.mm(beta).t() + repeated_mu0  # [D X N]
        # O(n^3)
        hBLK2 = cinvBs.bmm(Kg.unsqueeze(
            0).repeat_interleave(self.D, dim=0))  # [D X N X N]
        # O(n^3) (will need the full matrix for derivatives)
        Sigma = Kg - hBLK2.transpose(1, 2).bmm(hBLK2)  # [D X N X N]

        R = torch.exp(mu - 0.5 * torch.diagonal(Sigma,
                                                dim1=1, dim2=2))  # [D X N]
        p = 1e-3  # This value doesn't affect the result, it is a scaling factor
        scale = (1 / torch.sqrt(p + R)).unsqueeze(2)  # [D X N X 1]
        Rscale = (1 / (1 + p / R))    # [D X N]

        # O(n^3)
        Ls = torch.cholesky(
            Kf.mul(scale.bmm(scale.transpose(1, 2))) +
            Rscale.diag_embed(dim1=1),
            # torch.eye(self.N),
            upper=True)  # [D X N X N]
        Lys, _ = torch.solve(
            (self.Y.t() * scale.squeeze(2)).unsqueeze(2), Ls.transpose(1, 2))
        alphascale, _ = torch.solve(Lys, Ls)
        alphascale = alphascale.squeeze(2)
        alpha = (alphascale * (scale.squeeze(2))).t()  # [N X D]

        self.scale = scale
        self.Ls = Ls
        self.cinvBs = cinvBs
        self.alpha = alpha
        self.beta = beta

    def Marginalized_Variational_Bound(self, n_batch=None):
        '''
        alpha:[N X D]     | beta  :[N X D]   | mu0:[D]
        scale:[D X N X 1] | KXx   :[N X N*]  | Kxx:[N* X N*]
        Ls   :[D X N X N] | cinvBs:[D X N X N]
        '''
        if n_batch is None:
            ind = torch.arange(self.N)
        else:
            ind = torch.randperm(self.N)[:n_batch]
        N = len(ind)

        Kf = self.fkern.K(self.X[ind])
        Kg = self.gkern.K(self.X[ind])
        Lambda = torch.exp(self.Lambda[:, ind])         # [D X N]
        mu0 = torch.exp(self.mu0)               # [D]

        # mu_g
        beta = (Lambda - 0.5).t()  # [N X D]
        repeated_mu0 = mu0[:, None].repeat_interleave(
            N, dim=1)  # [D X N]
        mu = Kg.mm(beta).t() + repeated_mu0  # [D X N]

        # Sigma_g
        sLambda = torch.sqrt(Lambda).unsqueeze(2)  # [D X N X 1]
        Kgscaled = Kg.mul(sLambda.bmm(
            sLambda.transpose(1, 2)))  # [D X N X N]
        cinvB, LU = torch.solve(torch.eye(N), torch.cholesky(
            torch.eye(N) + Kgscaled, upper=False))  # [D X N X N]
        A = torch.ones(self.D, N, 1).bmm(sLambda.transpose(1, 2))
        cinvBs = cinvB.mul(A)
        # O(n^3)
        hBLK2 = cinvBs.bmm(Kg.unsqueeze(
            0).repeat_interleave(self.D, dim=0))  # [D X N X N]
        # O(n^3) (will need the full matrix for derivatives)
        Sigma = Kg - hBLK2.transpose(1, 2).bmm(hBLK2)  # [D X N X N]

        # R
        p = 1e-3  # This value doesn't affect the result, it is a scaling factor
        R = torch.exp(mu - 0.5 * torch.diagonal(Sigma,
                                                dim1=1, dim2=2))  # [D X N]
        scale = (1 / torch.sqrt(p + R)).unsqueeze(2)  # [D X N X 1]
        Rscale = (1 / (1 + p / R))    # [D X N]

        # O(n^3)
        Ls = torch.cholesky(
            Kf.mul(scale.bmm(scale.transpose(1, 2))) +
            Rscale.diag_embed(dim1=1),
            upper=True)  # [D X N X N]
        Lys, _ = torch.solve(
            (self.Y[ind].t() * scale.squeeze(2)).unsqueeze(2), Ls.transpose(1, 2))
        alphascale, _ = torch.solve(Lys, Ls)
        alphascale = alphascale.squeeze(2)
        alpha = (alphascale * (scale.squeeze(2))).t()  # [N X D]
        # term1: logN(y|0,Kf+R)
        term1 = -0.5 * (self.Y[ind].t().mm(alpha).diag()).sum() \
                - (torch.log(torch.diagonal(Ls, dim1=1, dim2=2))).sum()\
            + (torch.log(scale)).sum() \
            - 0.5 * N * torch.log(2 * torch.tensor(np.pi))
        # term2: -KL(N(g|mu,Sigma)||N(g|0,Kg))
        term2 = -0.5 * (beta.t().mm((mu - repeated_mu0).t()).diag()).sum() \
            + (torch.log(torch.diagonal(cinvB, dim1=1, dim2=2))).sum()\
            - 0.5 * (cinvB ** 2).sum() + N / 2
        # term3: Normalization
        term3 = -0.25 * torch.diagonal(Sigma, dim1=1, dim2=2).sum()

        F = term1 + term2 + term3

        return -F

    def compute_grad(self, flag):
        self.Lambda.requires_grad = flag
        self.fkern.compute_grad(flag)
        self.gkern.compute_grad(flag)
        self.mu0.requires_grad = flag

        self.hyperparameters = {'Lambda': self.Lambda, 'fkern': self.fkern.param(),
                                'gkern': self.gkern.param(), 'mu0': self.mu0}

    def learning(self, max_iter=100, n_batch=None):
        '''
        Optimizer
        - Adam
        - LBFGS
            This is a very memory intensive optimizer
            (it requires additional param_bytes * (history_size + 1) bytes).
            If it doesn’t fit in memory
            try reducing the history size, or use a different algorithm.
            < OH >
            it did't affected from scale of output compare to adam
        '''
        self.compute_grad(True)
        param = [self.Lambda] + self.fkern.param() + \
            self.gkern.param() + [self.mu0]
        # optimizer = torch.optim.Adam(param, lr=1e-4)
        optimizer = torch.optim.Adagrad(param)
        # optimizer = torch.optim.Adadelta(param, lr=0.04)
        # optimizer = torch.optim.LBFGS(
        #     param, lr=1e-2, history_size=50, line_search_fn='strong_wolfe')

        for i in range(max_iter):
            if optimizer.__class__.__name__ == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    f = self.Marginalized_Variational_Bound(n_batch=n_batch)
                    f.backward()
                    return f
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                f = self.Marginalized_Variational_Bound(n_batch=n_batch)
                f.backward()
                optimizer.step()
            for key in self.hyperparameters:
                if key == 'gkern' or key == 'fkern':
                    print(key + ' : ', torch.exp(self.hyperparameters[key][0]))
                else:
                    print(key + ' : ', torch.exp(self.hyperparameters[key]))

        self.compute_grad(False)
        # print(self.Marginalized_Variational_Bound())
        self.Precomputation()

    def predict(self, x):
        '''
        alpha:[N X D]     | beta  :[N X D]   | mu0:[D]
        scale:[D X N X 1] | KXx   :[N X N*]  | Kxx : [N* X N*]
        Ls   :[D X N X N] | cinvBs:[D X N X N]
        '''
        Kfxx, KfXx = self.fkern.K(x), self.fkern.K(
            self.X, x)    # test covariance f
        Kgxx, KgXx = self.gkern.K(x), self.gkern.K(
            self.X, x)    # test covariance g

        fmean = KfXx.t().mm(self.alpha)                          # predicted mean  f
        gmean = KgXx.t().mm(self.beta) + torch.exp(self.mu0)     # predicted mean  g
        ymean = fmean                                            # predicted mean  y

        # # Variance---------
        v, _ = torch.solve(
            ((self.scale * torch.ones(1, x.shape[0])).mul(KfXx)),
            self.Ls.transpose(1, 2))  # [D X N X N*]
        # predicted variance f
        diagCtst = (torch.diag(Kfxx) - (v * v).sum(1)).t()  # [N* X D]

        v = self.cinvBs.matmul(KgXx)  # [D X N X N*]
        # predicted variance g
        diagSigmatst = (torch.diag(Kgxx) - (v * v).sum(1)).t()  # [N* X D]
        # predicted variance y
        yvar = diagCtst + torch.exp(gmean + diagSigmatst * 0.5)

        return ymean, yvar


if __name__ == "__main__":
    from kernel import GaussianKernel
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    # Make toy dataset
    N = 600
    X = torch.linspace(0, np.pi * 2, N)[:, None]

    def true(X):
        # return torch.cos(X)
        return torch.sin(X)
    # Y = true(X) + torch.randn(N)[:, None] * 0.01
    Y = []
    for x in X:
        var = torch.normal(mean=0, std=x * x)[0]
        y = x * x + var
        Y.append(y)
    Y = torch.cat(Y)[:, None]
    Y /= Y.max()

    fkern = GaussianKernel()
    gkern = GaussianKernel()
    model = VHGPR(X, Y, fkern, gkern)
    model.learning(max_iter=300, n_batch=None)

    X = X.numpy().ravel()
    Y = Y.numpy().ravel()

    # Test data
    xx = torch.linspace(min(X), max(X), 100)[:, None]
    mm, vv = model.predict(xx)

    mm = mm.numpy().ravel()
    ss = np.sqrt(vv.numpy().ravel())
    xx = xx.numpy().ravel()

    plt.figure(figsize=(10, 5))

    line = plt.plot(xx, mm, label='Learned Policy',
                    linewidth=1, color='#348ABD')
    plt.fill_between(xx, mm + ss, mm - ss,
                     color=line[0].get_color(), alpha=0.2)
    point = plt.plot(X, Y, "*", markersize=5,
                     label='Training Set', color='#E24A33')

    plt.xlabel('X', fontsize=30)
    plt.ylabel('Y', fontsize=30)

    leg = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=4,
                     mode='expand', loc='lower left', fontsize=15)

    plt.show()
