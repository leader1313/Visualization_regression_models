import matplotlib.pyplot as plt
import torch
import numpy as np


class MultimodalHeteroToy():
    def __init__(self, N=100, M=2, name=str):
        # Make toy dataset
        self.N = N  # number of data set
        self.M = M  # number of modality
        self.X_train = torch.linspace(0, np.pi * 2, self.N)[:, None]
        if name == 'simple':
            self.L = 1e2  # distance between functions
            self.Y_train, self.mm_true, self.ss_true = self.simple_true(
                self.X_train, M=self.M, L=self.L)
        elif name == 'complex':
            self.B = 1  # number of bump
            self.Y_train, self.mm_true, self.ss_true = self.complex_true(
                self.X_train, M=self.M, B=self.B)
        self.X_train = torch.stack(
            [self.X_train for m in range(M)]).float().squeeze()

    def normalization(self, Y, mean, std):
        # normalize
        y_max = Y.abs().max(1)[0]
        norm_coef = (1 / y_max).repeat(self.N, 1).T  # M X N
        mean *= norm_coef
        std = std.squeeze() * norm_coef
        Y *= norm_coef

        return Y, mean, std

    def simple_true(self, X, M=2, L=1e2):
        N = X.shape[0]
        mean = torch.linspace(-1, 1, M + 2)[1:-1]
        mean = torch.stack([(mean[m] + (mean[m] * L * X)).squeeze()
                           for m in range(M)])
        std = X ** 2
        var = torch.normal(mean=torch.zeros(N), std=std).diag()
        Y = mean + var

        return self.normalization(Y, mean, std)

    def complex_true(self, X, M=2, B=2):
        N = X.shape[0]
        mean = torch.linspace(-1, 1, M + 2)[1:-1]
        mean = torch.stack([(3 * X * mean[m] + torch.cos(X)).squeeze()
                            for m in range(M)])
        std = torch.sin(X * np.pi * B / 6) ** 2 * 2
        var = torch.normal(mean=torch.zeros(N), std=std).diag()
        Y = mean + var

        return self.normalization(Y, mean, std)


if __name__ == "__main__":
    toy = MultimodalHeteroToy(name="simple")
    # toy = MultimodalHeteroToy(name="complex")
    X, Y = toy.X_train, toy.Y_train
    mm, ss = toy.mm_true, toy.ss_true

    for m in range(toy.M):
        sca = plt.scatter(X[m], Y[m], marker="+", s=50, c='tomato')
        line = plt.plot(X[m], mm[m], color='black', linewidth=2)
        plt.fill_between(
            X[m],
            mm[m] - ss[m],
            mm[m] + ss[m],
            alpha=0.3,
            # color=line[0].get_color(),
            color='green',
        )
    plt.xticks(torch.linspace(X.min(), X.max(), steps=5))
    plt.yticks(torch.linspace(Y.min(), Y.max(), steps=5))

    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    plt.show()
