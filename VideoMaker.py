from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
from os.path import isfile, join
import re


class VideoMaker():
    def __init__(self, model=object, video_name='video_name'):
        now = datetime.now(timezone(timedelta(hours=9)))
        yd_string = now.strftime('%Y%m%d')
        time_string = now.strftime('%H%M%S')
        dir_ID = video_name + '/' + yd_string
        self.figure_dir = 'Figures/' + dir_ID + '/' + time_string + '/'
        self.video_dir = 'Videos/' + dir_ID
        dirs = {'fig': self.figure_dir, 'video': self.video_dir}
        for key in dirs:
            if not os.path.exists(dirs[key]):
                os.makedirs(dirs[key])

        self.file_name = time_string
        self.model = model

    # snapshots from learning process -------

    def snap_shot(self, step, filling=False):
        plt.style.use("ggplot")
        X_train, Y_train = self.model.X, self.model.Y
        xx = torch.linspace(X_train.min(), X_train.max(),
                            steps=100).unsqueeze(1)
        # xx = np.linspace(min(X_train), max(X_train), 100)
        # xx = torch.from_numpy(xx).float()
        try:
            # homoscedastic
            mm, vv = self.model.predict(xx)
            noise = torch.exp(self.model.log_sigma[-1])
            noise = torch.sqrt(noise)
        except:
            # heteroscedastic
            mm, vv, ygvar = self.model.predict(xx)
            noise = torch.sqrt(ygvar)
        ss = torch.sqrt(vv)

        plt.scatter(X_train, Y_train, marker="*", s=100)
        plt.xlim(min(X_train), max(X_train))
        plt.ylim(min(Y_train), max(Y_train))
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)

        for m in range(self.model.M):
            line = plt.plot(xx, mm[m])
            if filling:
                # plt.fill_between(
                #     xx.squeeze(),
                # mm[m, :, 0] + ss[m, 0],
                # mm[m, :, 0] - ss[m, 0],
                #     color=line[0].get_color(),
                #     alpha=0.1,
                # )
                plt.fill_between(
                    xx.squeeze(),
                    mm[m, :, 0] + noise.squeeze(),
                    mm[m, :, 0] - noise.squeeze(),
                    color="blue",
                    alpha=0.2,
                    hatch="\\",
                )
        plt.savefig(self.figure_dir + 'frame' + str(step))
        plt.clf()

    def make_figs(self, max_n=10):
        self.snap_shot(0, filling=True)
        for i in range(max_n):
            self.model.learning(max_iter=1)
            self.snap_shot(i + 1, filling=True)
            done = int(i / (max_n - 1) * 10)
            bar = u"\u2588" * done + ' ' * (10 - done)
            per = (i + 1) * 100 / max_n
            print('\r[{}] {} %'.format(bar, per), end='')
        self.snap_shot(max_n + 1, filling=True)

    # figures to video ----------------------
    def tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def figure2video(self, fps=10):
        pathIn = self.figure_dir
        pathOut = self.video_dir + self.file_name + '.mp4'

        frame_array = []
        # files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and ]
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))
                 and not f.startswith('.')]
        files.sort(key=self.alphanum_key)  # sorting the file names properly

        for i in range(len(files)):
            filename = pathIn + files[i]
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)

            # inserting the frames into an image array
            frame_array.append(img)
        # fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(pathOut, fourcc, fps, size)
        # writing to a image array
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        print('\nCompleted to make VIDEO: ' + self.file_name + '.mp4')


if __name__ == '__main__':
    from Models import IOMHGP, GaussianKernel
    # Import toy data
    N = 20
    # X_limit = [-torch.tensor(np.pi),torch.tensor(np.pi)]
    X_limit = [-torch.tensor(3.0), torch.tensor(3.0)]
    X = torch.linspace(min(X_limit), max(X_limit), N)[:, None]
    Y1 = torch.sin(X) + torch.randn(N)[:, None] * 0.15
    Y2 = torch.cos(X) + torch.randn(N)[:, None] * 0.15
    K = torch.tensor([])
    fkern = GaussianKernel()
    gkern = GaussianKernel()

    # Import learning model
    model = IOMHGP(
        torch.cat([X, X]).float(),
        torch.cat([Y1, Y2]).float(),
        fkern, gkern,
        M=2,
    )

    # Video making
    videoMaker = VideoMaker(model=model, video_name='test')
    videoMaker.make_figs(max_n=30)
    videoMaker.figure2video()
