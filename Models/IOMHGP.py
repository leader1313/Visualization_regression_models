import torch
import numpy as np
import copy

"""
To do
1) LBFGS parameter surch
2) add iomgp parts
    - multiple f kernel
    - hyperparameters
        * beta
        *
"""


class IOMHGP:
    def __init__(self, X, Y, fkernel, gkernel, M=2, lengthscale=1.0):
        """
        IOMHGP - Infinite Mixtures of Heteroscedastic GPs
            Input:
                - X[N X Q] : Training input data. One vector per row.
                - y[N X D] : Training output value. One scalar per row.
                - fkernel  : Covariance function for the GP f (signal).
                - gkernel  : Covariance function for the GP g (noise).
                - M        : Upper bound of mixtures
                - N        : Size of dataset
                - D        : Dimension of Y
                - Hyperparameters = {Lamda[D X N], hyperf[1], hyperg[1], mu0[D]}
        Copyright (c) 2021 by Hanbit Oh
        """
        self.X = X
        self.Y = Y
        self.N, self.D = self.Y.shape
        self.M = M

        self.MVB = 1e9
        self.p_mixture_thresh = self.N * (1 / self.M) * 0.1
        self.p_instance_thresh = 1e-4
        # Hyperparameters Initialization
        # Hyper of kernel
        # Lengthscale : affect for precise allocation
        lengthscales = torch.log((self.X.max() - self.X.min()) *
                                 lengthscale).unsqueeze(0)
        f_param = lengthscales
        g_param = lengthscales

        self.fkern = []
        for m in range(self.M):
            fkernel.__init__(param=f_param)
            self.fkern += [copy.deepcopy(fkernel)]
        self.fkern = np.array(self.fkern)  # f: policy function
        gkernel.__init__(param=g_param)
        self.gkern = copy.deepcopy(gkernel)  # g: noise function
        # Hyper of g
        # SignalPower = torch.var(self.Y, 0)
        # NoisePower = 0.125 * SignalPower
        NoisePower = torch.tensor(0.01) * torch.ones(self.D)
        self.Lambda = torch.log(torch.tensor(0.5)) * torch.ones(self.D, self.N)
        self.mu0 = torch.log(NoisePower)  # [D]
        # Hyper of v
        self.v_beta_0 = torch.log(torch.tensor(100.0))

        # Prior initialization
        self.q_z_pi = torch.ones(self.M, self.N) / self.M  # [M X N]

        self.v_alpha = torch.ones(self.M)  # [M]
        self.v_beta = torch.ones(self.M) * torch.exp(self.v_beta_0)  # [M]

        self.q_f_mean = torch.tensor(
            np.random.normal(0, 0.01, (self.M, self.N, self.D))
        ).float()  # [M X N X D]
        self.q_f_sig = torch.stack(
            [
                torch.stack([torch.eye(self.N) for m in range(self.M)])
                for d in range(self.D)
            ]
        )  # [D X M X N X N]
        self.q_g_mean = torch.exp(self.mu0).repeat(self.N, 1)  # [N X D]
        self.q_g_sig = torch.stack(
            [torch.eye(self.N) for d in range(self.D)]
        )  # [D X N X N]

        self.hyperparameters = {
            "Lambda": self.Lambda,
            "fkern": [self.fkern[m].param() for m in range(self.M)],
            "gkern": self.gkern.param(),
            "mu0": self.mu0,
            "v_beta_0": self.v_beta_0,
        }
        self.expectation(max_iter=30)
        # self.update_q_g()
        # self.Marginalized_Variational_Bound()
        self.maximization(max_iter=10, type2_l=True, type2_h=True)

    def precomputation(self, batch_ind=None):
        if batch_ind is None:
            ind = torch.arange(self.N)
        else:
            ind = batch_ind

        Kf = torch.stack(
            [
                self.fkern[m].K(self.X[ind]) + torch.eye(ind.shape[0]) * 1e-5
                for m in range(self.M)
            ]
        )  # [M X N' X N']
        Psi = 1 / torch.exp(
            self.q_g_mean - 0.5 * self.q_g_sig.diagonal(dim1=1, dim2=2).t()
        )  # [N X D]
        Psi = Psi[ind, :]  # [N' X D]
        B = self.q_z_pi[:, ind].matmul(
            Psi.T.diag_embed(dim1=1))  # [D X M X N']

        p = 1e-3  # This value doesn't affect the result, it is a scaling factor
        scale = (torch.sqrt(p + B)).unsqueeze(3)  # [D X M X N' X 1]
        pB = copy.deepcopy(B.detach())
        pB[B > p] = p / pB[B > p]
        Bscale = 1 + pB  # [D X M X N']

        # O(n^3)
        Ls = torch.cholesky(
            Kf.mul(scale.matmul(scale.transpose(2, 3))) +
            Bscale.diag_embed(dim1=2),
            upper=True,
        )  # [D X M X N' X N']

        Lys, _ = torch.solve(
            (self.Y[ind].repeat(self.M, 1, 1).permute(2, 0, 1) *
             scale.squeeze(3)).unsqueeze(3),
            Ls.transpose(2, 3),
        )  # [D X M X N' X 1]

        alphascale, _ = torch.solve(Lys, Ls)  # [D X M X N' X 1]
        alphascale = alphascale.squeeze(3)  # [D X M X N']
        alpha = (alphascale * (scale.squeeze(3))
                 ).permute(1, 2, 0)  # [M X N' X D]

        if batch_ind is None:
            self.Kf = Kf
            self.B = B
            self.scale = scale
            self.Ls = Ls
            self.alpha = alpha

        return B, scale, Ls, alpha

    def update_q_z(self):

        E_ln_v = torch.digamma(self.v_alpha) - torch.digamma(
            self.v_alpha + self.v_beta
        )  # [M]
        E_ln_1_minus_v = torch.digamma(self.v_beta) - torch.digamma(
            self.v_alpha + self.v_beta
        )  # [M]

        digamma_sum = torch.zeros(self.M)  # [M]
        for m in range(0, self.M):
            digamma_sum[m] += E_ln_v[m]
            for i in np.arange(0, m):
                digamma_sum[m] += E_ln_1_minus_v[i]

        Psi = torch.exp(
            self.q_g_mean - 0.5 * self.q_g_sig.diagonal(dim1=1, dim2=2).t()
        )  # [N X D]

        ln_rho = (
            -0.5
            * (
                (
                    (
                        (self.Y.repeat(self.M, 1, 1) - self.q_f_mean) ** 2
                        + torch.diagonal(self.q_f_sig, dim1=2,
                                         dim2=3).permute(1, 2, 0)
                    )
                    / Psi
                ).sum(2)
                + torch.log(np.pi * 2 * Psi).repeat(self.M, 1, 1).sum(2)
            )
            + digamma_sum.repeat(self.N, 1).T
        )  # [M X N]
        ln_rho -= ln_rho.max(0)[0]

        self.q_z_pi = torch.exp(ln_rho)  # [M X N]
        self.q_z_pi /= self.q_z_pi.sum(0)[None, :]
        # self.q_z_pi[torch.isnan(self.q_z_pi)] = 1.0 / self.M

        self.digamma_sum = digamma_sum
        self.E_ln_v = E_ln_v
        self.E_ln_1_minus_v = E_ln_1_minus_v

    def update_q_v(self):
        v_beta_0 = torch.exp(self.v_beta_0)  # [M]
        for m in range(0, self.M):
            self.v_alpha[m] = 1.0 + self.q_z_pi[m, :].sum()  # [M]
            tmpSum = torch.zeros(1)
            for j in range(m + 1, self.M):
                tmpSum += self.q_z_pi[j, :].sum()

            self.v_beta[m] = v_beta_0 + tmpSum  # [M]

    def update_q_f(self):
        B, scale, Ls, alpha = self.precomputation()
        self.q_f_mean = self.Kf.transpose(1, 2).bmm(alpha)  # [M X N X D]

        v, _ = torch.solve(
            ((scale.repeat(1, 1, 1, self.N)).mul(self.Kf)), Ls.transpose(2, 3)
        )  # [D X M X N X N]
        self.q_f_sig = self.Kf - v.transpose(2, 3).matmul(v)  # [D X M X N X N]

    def update_q_g(self, batch_ind=None):
        if batch_ind is None:
            ind = torch.arange(self.N)
        else:
            ind = batch_ind

        N = len(ind)  # [N']
        Kg = self.gkern.K(self.X[ind])  # [D X N' X N']
        Lambda = torch.exp(self.Lambda[:, ind])  # [D X N']
        mu0 = torch.exp(self.mu0)  # [D]

        sLambda = torch.sqrt(Lambda).unsqueeze(2)  # [D X N' X 1]
        Kgscaled = Kg.mul(sLambda.bmm(
            sLambda.transpose(1, 2)))  # [D X N' X N']
        cinvB, LU = torch.solve(
            torch.eye(N), torch.cholesky(torch.eye(N) + Kgscaled, upper=False)
        )  # [D X N' X N']
        A = torch.ones(self.D, N, 1).bmm(
            sLambda.transpose(1, 2))  # [D X N' X N']
        cinvBs = cinvB.mul(A)  # [D X N' X N']
        beta = (Lambda - 0.5).t()  # [N' X D]

        N_mu0 = mu0.repeat(N, 1)  # [N' X D]

        self.q_g_mean[ind, :] = Kg.mm(beta) + N_mu0  # [N' X D]
        # O(n^3)
        hBLK2 = cinvBs.bmm(
            Kg.unsqueeze(0).repeat_interleave(self.D, dim=0)
        )  # [D X N' X N']
        # O(n^3) (will need the full matrix for derivatives)

        sigma = copy.deepcopy(self.q_g_sig.diagonal(
            dim1=1, dim2=2).detach())  # [D X N]
        sigma[:, ind] = (Kg - hBLK2.transpose(1, 2).bmm(hBLK2)
                         ).diagonal(dim1=1, dim2=2)  # [D X N']
        self.q_g_sig = sigma.diag_embed(dim1=1)  # [D X N X N]
        # self.q_g_sig = Kg - hBLK2.transpose(1, 2).bmm(hBLK2)  # [D X N X N]
        self.cinvBs = cinvBs
        self.beta = beta

        return cinvB, beta, N_mu0

    def expectation(self, max_iter=10):
        for _ in range(max_iter):
            with torch.no_grad():
                self.update_q_f()
                self.update_q_v()
                self.update_q_z()
            # self.maximization(max_iter=1, type2_l=True)

    def Marginalized_Variational_Bound(self, n_batch=None):
        """
        cinvB:[D X N' X N']|beta:[N' X D]         | N_mu0 : [N' X D]
        B    :[D X M X N'] |scale:[D X M X N' X 1]| Ls    :[D X M X N' X N']
        alpha:[M X N' X D]
        """
        if n_batch is None:
            ind = torch.arange(self.N)
        else:
            ind = torch.randperm(self.N)[:n_batch]
        N = len(ind)

        cinvB, beta, N_mu0 = self.update_q_g(batch_ind=ind)
        B, scale, Ls, alpha = self.precomputation(batch_ind=ind)

        # term1: logN(a*|0,Kf+B^(-1))
        # We won't update term1 if the current mixture under
        # consideration has no instances associated with it
        m_ind = self.q_z_pi.sum(1) > self.p_mixture_thresh
        M = m_ind.sum()
        scale = scale[:, m_ind].squeeze(3)
        Ls = Ls[:, m_ind]
        # i_ind = self.q_z_pi[:, ind] > self.p_instance_thresh
        # i_scale = scale[:, i_ind]
        # i_Ls = torch.diagonal(Ls, dim1=2, dim2=3)[:, i_ind]

        term1 = (
            -0.5
            * (
                self.Y[ind]
                .repeat(M, 1, 1)
                .permute(0, 2, 1)
                .bmm(alpha[m_ind])
                .diagonal(dim1=1, dim2=2)
            ).sum()
            - (torch.log(torch.diagonal(Ls, dim1=2, dim2=3))).sum()
            + (torch.log(scale)).sum()
            - 0.5 * M * N * self.D * torch.log(2 * torch.tensor(np.pi))
            # - (torch.log(i_Ls)).sum()
            # + (torch.log(i_scale)).sum()
            # - 0.5 * i_ind.ravel().shape[0] * self.D *
            # torch.log(2 * torch.tensor(np.pi))
        )

        # term2: -KL(N(g|mu,Sigma)||N(g|0,Kg))
        term2 = (
            -0.5 * beta.t().mm(self.q_g_mean[ind] - N_mu0).diag().sum()
            + (torch.log(torch.diagonal(cinvB, dim1=1, dim2=2))).sum()
            - 0.5 * (cinvB ** 2).sum()
            + N / 2
        )

        # term3: Normalization

        # B = B[B > 1e-2]
        term3 = -0.5 * (
            self.D
            * (((self.q_z_pi[m_ind][:, ind] - 1) * torch.log(2 * torch.tensor(np.pi))).sum())
            + (
                self.q_z_pi[m_ind][:, ind]
                .repeat(self.D, 1, 1)
                .permute(1, 2, 0)
                .mul(self.q_g_mean[ind].repeat(M, 1, 1))
            ).sum()
            + torch.log(B[:, m_ind] + 1e-2).sum()
            # + torch.log(B).sum()
        )

        # term4: E_zv(p(z|v))
        r = copy.deepcopy(self.q_z_pi[m_ind][:, ind])
        r[r != 0] = torch.log(r[r != 0])

        term4 = (
            self.q_z_pi[m_ind][:, ind].mul(
                self.digamma_sum[m_ind].repeat(N, 1).T).sum()
            - self.q_z_pi[m_ind][:, ind].mul(r).sum()
        )

        # term5: KL(v*|v)
        v_beta_0 = self.v_beta_0.exp()
        term5 = (
            -(
                torch.lgamma(self.v_alpha[m_ind])
                + torch.lgamma(self.v_beta[m_ind])
                - torch.lgamma(self.v_alpha[m_ind] + self.v_beta[m_ind])
            ).sum()
            + M * (torch.lgamma(v_beta_0) - torch.lgamma(1 + v_beta_0))
            + (self.v_alpha[m_ind] - 1).mul(self.E_ln_v[m_ind]).sum()
            + (self.v_beta[m_ind] -
               v_beta_0).mul(self.E_ln_1_minus_v[m_ind]).sum()
        )

        F = term1 + term2 + term3 + term4 + term5
        return -F

    def save_checkpoint(self):
        """
        save checkpoint for load optimized parameters when we need it
        """
        v_beta_0 = [self.v_beta_0]
        mu0 = self.mu0
        Lambda = self.Lambda
        q_z_pi = self.q_z_pi
        gkern = self.gkern.param()
        fkerns = []
        for m in range(self.M):
            fkerns += self.fkern[m].param()

        torch.save(
            {
                "beta0": v_beta_0,
                "mu0": mu0,
                "Lambda": Lambda,
                "q_z_pi": q_z_pi,
                "gkern": gkern,
                "fkerns": fkerns,
            },
            "checkpoint.pt",
        )

    def load_checkpoint(self):
        """
        load checkpoint for load optimized parameters when we need it
        """
        checkPoint = torch.load("checkpoint.pt")

        self.q_z_pi = checkPoint["q_z_pi"]
        self.v_beta_0 = checkPoint["beta0"][0]
        self.mu0 = checkPoint["mu0"]
        self.Lambda = checkPoint["Lambda"]
        self.gkern.LengthScales = checkPoint["gkern"][0]
        fkerns = checkPoint["fkerns"]
        for m in range(self.M):
            self.fkern[m].LengthScales = fkerns[m]
        return checkPoint

    def show_hyperparameters(self):
        print("-" * 20)
        for key in self.hyperparameters:
            if key == "gkern":
                print(key + " : ", torch.exp(self.hyperparameters[key][0]))
            elif key == "fkern":
                for m in range(len(self.hyperparameters[key])):
                    print(key, m + 1, " : ",
                          torch.exp(self.hyperparameters[key][m][0]))
            elif key == "Lambda":
                pass
            else:
                print(key + " : ", torch.exp(self.hyperparameters[key]))

    def compute_grad(self, type2_l=False, type2_h=False):
        self.Lambda.requires_grad = type2_l
        self.v_beta_0.requires_grad = type2_h
        self.mu0.requires_grad = type2_h
        self.gkern.compute_grad(type2_h)
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.p_mixture_thresh:
                self.fkern[m].compute_grad(type2_h)

    def maximization(self, max_iter=100, n_batch=None, type2_l=False, type2_h=False):
        """
        Optimizer
        - Adam
        - LBFGS
            This is a very memory intensive optimizer
            (it requires additional param_bytes * (history_size + 1) bytes).
            If it doesn’t fit in memory
            try reducing the history size, or use a different algorithm.
            < OH >
            it did't affected from scale of output compare to adam
        """
        self.compute_grad(type2_l=type2_l, type2_h=type2_h)
        param = [self.Lambda] + self.gkern.param() + [self.mu0] + \
            [self.v_beta_0]
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.p_mixture_thresh:
                param += self.fkern[m].param()

        # optimizer = torch.optim.Adam(param, lr=1e-4)
        # optimizer = torch.optim.Adam(param)
        # optimizer = torch.optim.Adagrad(param, lr=1e-2)
        optimizer = torch.optim.Adagrad(param, lr=5e-3)
        # optimizer = torch.optim.LBFGS(
        #     param, lr=1e-1, history_size=50, line_search_fn="strong_wolfe"
        # )

        for i in range(max_iter):
            if optimizer.__class__.__name__ == "LBFGS":

                def closure():
                    optimizer.zero_grad()
                    f = self.Marginalized_Variational_Bound(n_batch=n_batch)
                    f.backward(retain_graph=True)
                    return f

                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                f = self.Marginalized_Variational_Bound(n_batch=n_batch)
                f.backward(retain_graph=True)
                optimizer.step()
            # self.show_hyperparameters()

        self.compute_grad(type2_l=False, type2_h=False)
        self.precomputation()

    def learning(self, max_iter=10):
        NL = 1e9
        self.save_checkpoint()
        step = 0
        stop_flag = False
        Max_patient = 5
        patient_count = 0

        while (step < max_iter) and not (stop_flag):
            step += 1
            # print("=========================")
            # print("E step")
            self.expectation(max_iter=30)
            # print("M step")
            self.maximization(max_iter=10, type2_l=True, type2_h=True)
            # self.maximization(max_iter=100, n_batch=10,
            #                   type2_l=True, type2_h=True)
            # type-2 likelihood ----------
            # self.maximization(max_iter=3, type2_h=True)
            # self.show_hyperparameters()
            temp_bound = self.Marginalized_Variational_Bound()

            # print(step, " th Marginalized VB : ", temp_bound)
            # print("Z : ", self.q_z_pi.sum(axis=1))

            if NL > temp_bound:
                patient_count = 0
                NL = temp_bound
                self.save_checkpoint()

            else:
                patient_count += 1
                # print("-------Patient_Count(< %i) : %i" %
                #       (Max_patient, patient_count))
                if patient_count >= Max_patient:
                    stop_flag = True
                # import ipdb
                # ipdb.set_trace()

        self.load_checkpoint()

        # print(self.q_z_pi.sum(axis=1))
        m_ind = self.q_z_pi.sum(axis=1) > self.p_mixture_thresh
        N_Mixture = m_ind.sum()

        try:
            self.fkern = self.fkern[m_ind]
        except:  # M=1
            pass

        self.q_z_pi = self.q_z_pi[m_ind]
        (
            self.digamma_sum,
            self.v_alpha,
            self.v_beta,
            self.E_ln_v,
            self.E_ln_1_minus_v,
        ) = (
            self.digamma_sum[m_ind],
            self.v_alpha[m_ind],
            self.v_beta[m_ind],
            self.E_ln_v[m_ind],
            self.E_ln_1_minus_v[m_ind],
        )
        self.M = N_Mixture
        self.precomputation()
        # print("-------------------------------------------")
        # print("Number of Mixture : %i" % (N_Mixture))

    def predict(self, x):
        """
        alpha:[M X N X D]     | beta  :[N X D]        | mu0:[D]
        scale:[D X M X N X 1] | Ls   :[D X M X N X N] | cinvBs:[D X N X N]
        KfXx   :[M X N X N*]  | Kfxx:[M X N* X N*]
        KgXx   :[N X N*]      | Kgxx:[N* X N*]
        """
        # test covariance f
        Kfxx = torch.stack([self.fkern[m].K(x) for m in range(self.M)])
        KfXx = torch.stack([self.fkern[m].K(self.X, x) for m in range(self.M)])
        # test covariance g
        Kgxx, KgXx = self.gkern.K(x), self.gkern.K(self.X, x)

        # Mean------------------------------------------------
        # predicted mean  f[M X N* X D]
        fmean = KfXx.transpose(1, 2).bmm(self.alpha)
        # predicted mean  g[N* X D]
        gmean = KgXx.t().mm(self.beta) + torch.exp(self.mu0)
        ymean = fmean  # predicted mean  y

        # Variance------------------------------------------------
        v, _ = torch.solve(
            (self.scale.repeat(1, 1, 1, x.shape[0])).mul(
                KfXx), self.Ls.transpose(2, 3)
        )  # [D X M X N X N*]
        # predicted variance f
        diagCtst = (torch.diagonal(Kfxx, dim1=1, dim2=2) - (v * v).sum(2)).permute(
            1, 2, 0
        )  # [M X N* X D]

        v = self.cinvBs.matmul(KgXx)  # [D X N X N*]
        # predicted variance g
        diagSigmatst = (torch.diag(Kgxx) - (v * v).sum(1)).t()  # [N* X D]
        # predicted variance y [M X N* X D]
        ygvar = torch.exp(gmean + diagSigmatst * 0.5)
        yvar = diagCtst + ygvar

        return ymean.detach(), yvar.detach(), ygvar.detach()


if __name__ == "__main__":
    from kernel import GaussianKernel
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")

    # Make toy dataset
    N = 300
    X = torch.linspace(0, np.pi * 2, N)[:, None]

    def true(X):
        # return torch.cos(X)
        return torch.sin(X)

    Y = []
    for x in X:
        var = torch.normal(mean=0, std=x * x)[0]
        y = x * x + var
        Y.append(y)
    Y = torch.cat(Y)[:, None]
    Y /= Y.max()

    fkern = GaussianKernel()
    gkern = GaussianKernel()
    model = IOMHGP(X, Y, fkern, gkern, M=1)
    model.learning(max_iter=10)

    X = X.numpy().ravel()
    Y = Y.numpy().ravel()
    M = model.M

    # Test data
    xx = torch.linspace(min(X), max(X), 100)[:, None]
    mean, var, _ = model.predict(xx)

    mean = mean.detach().numpy()
    var = np.sqrt(var.detach().numpy())
    xx = xx.numpy().ravel()

    plt.figure(figsize=(10, 5))

    for m in range(M):
        mm = mean[m].ravel()
        vv = var[m].ravel()
        line = plt.plot(xx, mm, label="Learned Policy",
                        linewidth=1, color="#348ABD")
        plt.fill_between(xx, mm + vv, mm - vv,
                         color=line[0].get_color(), alpha=0.2)

    data_point = plt.plot(
        X, Y, "*", markersize=5, label="Training Set", color="#E24A33"
    )
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    plt.show()
