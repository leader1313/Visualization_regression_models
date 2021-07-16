from Models import IOMHGP, GaussianKernel
from ToyPrograms import MultimodalHeteroToy
from VideoMaker import VideoMaker
import matplotlib.pyplot as plt
import torch


def main():
    # Import toy data
    toy = MultimodalHeteroToy(N=100, M=2, name="simple")
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

    fkern = GaussianKernel()
    gkern = GaussianKernel()

    # Import learning model
    model = IOMHGP(
        X.view(-1).unsqueeze(1),
        Y.view(-1).unsqueeze(1),
        fkern, gkern,
        M=10,
        lengthscale=2.0,
    )

    # Video making
    videoMaker = VideoMaker(model=model, video_name='Simple_Multimodal_hetero')
    videoMaker.make_figs(max_n=30)
    videoMaker.figure2video()


if __name__ == "__main__":
    main()
