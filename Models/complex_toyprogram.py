from kernel import GaussianKernel
from IOMHGP import IOMHGP
import matplotlib.pyplot as plt
import torch
import numpy as np


# plt.style.use("ggplot")

# Make toy dataset
N = 100
X_train = torch.linspace(0, np.pi * 2, N)[:, None]
M = 2

# Ground truth------------------------------


def true(X, M=2):
    N = X.shape[0]
    mean = torch.linspace(-1, 1, M + 2)[1:-1]
    mean = torch.stack([(3 * X * mean[m] + torch.cos(X)).squeeze()
                       for m in range(M)])
    # std = 1-((X-np.pi)/np.pi) ** 2
    std = torch.sin(X * 1.5) ** 2
    var = torch.normal(mean=torch.zeros(N), std=std).diag()
    # var /= var.max()
    Y = mean + var

    # normalize
    mean *= (1 / Y.abs().max(1)[0]).repeat(N, 1).T
    # std /= std.max()
    std = std.squeeze() * (1 / Y.abs().max(1)[0]).repeat(N, 1).T
    Y *= (1 / Y.abs().max(1)[0]).repeat(N, 1).T

    return Y, mean, std


Y_train, mm_true, ss_true = true(X_train, M=M)
for m in range(M):
    sca = plt.scatter(X_train, Y_train[m], marker="+", s=50, c='orange')
    line = plt.plot(X_train, mm_true[m], color='black', linewidth=2)
    plt.fill_between(
        X_train.squeeze(),
        mm_true[m] - ss_true[m],
        mm_true[m] + ss_true[m],
        alpha=0.3,
        # color=line[0].get_color(),
        color='purple',
    )
plt.xticks(torch.linspace(X_train.min(), X_train.max(), steps=5), alpha=0.0)
plt.yticks(torch.linspace(Y_train.min(), Y_train.max(), steps=5), alpha=0.0)

X_train = torch.cat([X_train for m in range(M)]).float()
Y_train = torch.cat([Y_train[m] for m in range(M)]).float().unsqueeze(1)

# Learning------------------------------
fkern = GaussianKernel()
gkern = GaussianKernel()
model = IOMHGP(X_train, Y_train, fkern, gkern, M=5)
model.learning(max_iter=10)

# Testing-------------------------------
M = model.M

# Test data
xx = torch.linspace(X_train.min(), X_train.max(), 100)[:, None]
mm_test, vv_test, _ = model.predict(xx)

mm_test = mm_test.detach()
ss_test = torch.sqrt(vv_test.detach())


for m in range(M):
    mm = mm_test[m].squeeze()
    ss = ss_test[m].squeeze()
    line = plt.plot(xx, mm, label="Learned Policy",
                    linewidth=1)  # , color="#348ABD")
    plt.fill_between(
        xx.squeeze(), mm + ss, mm - ss, color=line[0].get_color(), alpha=0.4
    )

plt.xlabel("X", fontsize=10)
plt.ylabel("Y", fontsize=10)

plt.show()
