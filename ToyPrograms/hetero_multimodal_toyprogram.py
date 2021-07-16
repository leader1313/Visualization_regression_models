import matplotlib.pyplot as plt
import torch
import numpy as np


class HeteroMultimodalToy():
    def __init__(self, N=100, M=2, B=2):
        # Make toy dataset
        self.N = N  # number of data set
        self.M = M  # number of modality
        self.B = B  # number of bump
        self.X_train = torch.linspace(0, np.pi * 2, self.N)[:, None]
        self.Y_train, self.mm_true, self.ss_true = self.true(
            self.X_train, M=self.M, B=self.B)
        self.X_train = torch.stack(
            [self.X_train for m in range(M)]).float().squeeze()

    def true(self, X, M=2, B=2):
        N = X.shape[0]
        mean = torch.linspace(-1, 1, M + 2)[1:-1]
        mean = torch.stack([(3 * X * mean[m] + torch.cos(X)).squeeze()
                            for m in range(M)])
        std = torch.sin(X * np.pi * B / 6) ** 2
        var = torch.normal(mean=torch.zeros(N), std=std).diag()
        Y = mean + var

        # normalize
        y_max = Y.abs().max(1)[0]
        norm_coef = (1 / y_max).repeat(N, 1).T  # M X N
        mean *= norm_coef
        std = std.squeeze() * norm_coef
        Y *= norm_coef
        return Y, mean, std


if __name__ == "__main__":
    toy = HeteroMultimodalToy()
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
    plt.xticks(torch.linspace(X.min(), X.max(), steps=5), alpha=0.0)
    plt.yticks(torch.linspace(Y.min(), Y.max(), steps=5), alpha=0.0)

    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    plt.show()
