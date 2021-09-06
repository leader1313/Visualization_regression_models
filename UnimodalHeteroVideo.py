from Models import IOMHGP, GaussianKernel, IOMGP
from ToyPrograms import MultimodalHeteroToy
from VideoMaker import VideoMaker
import matplotlib.pyplot as plt


class UnimodalHeteroVideo():
    def __init__(self):
        toy = MultimodalHeteroToy(N=100, M=1, name="complex")
        self.X, self.Y = toy.X_train, toy.Y_train
        self.mm, self.ss = toy.mm_true, toy.ss_true
        self.M = toy.M

    def hetero_model(self):
        # Import learning model
        fkern = GaussianKernel()
        gkern = GaussianKernel()
        model = IOMHGP(
            self.X.view(-1).unsqueeze(1),
            self.Y.view(-1).unsqueeze(1),
            fkern, gkern,
            M=1,
        )
        return model

    def homo_model(self):
        # Import learning model
        fkern = GaussianKernel()
        model = IOMGP(
            self.X.view(-1).unsqueeze(1),
            self.Y.view(-1).unsqueeze(1),
            fkern,
            Max_mixture=1,
            lengthscale=4.0,
        )
        return model

    def ground_truth(self):
        for m in range(self.M):
            line = plt.plot(self.X[m], self.mm[m], color='Green', linewidth=1)
            plt.fill_between(
                self.X[m],
                self.mm[m] - self.ss[m],
                self.mm[m] + self.ss[m],
                alpha=0.3,
                # color=line[0].get_color(),
                color='green',
            )

    def making_video(self):
        # Video making
        videoMaker = VideoMaker(
            model=self.hetero_model(), G=self.ground_truth, video_name='Unimodal/hetero')
        videoMaker.make_figs(max_n=30)
        videoMaker.figure2video()

        videoMaker = VideoMaker(
            model=self.homo_model(), G=self.ground_truth, video_name='Unimodal/homo')
        videoMaker.make_figs(max_n=30)
        videoMaker.figure2video()


if __name__ == "__main__":
    for _ in range(10):
        CMHV = UnimodalHeteroVideo()
        CMHV.making_video()
