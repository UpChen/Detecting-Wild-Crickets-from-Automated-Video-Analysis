import os
import cv2
import numpy as np
from tqdm import tqdm


class ViderParser:
    '''
    Video parser
    '''
    def __init__(self, video_path, frame_size=None):
        self.video_path = video_path
        self.video_basename = os.path.basename(self.video_path)
        print('Parsing the video {} ...'.format(os.path.abspath(video_path)))
        vcp = cv2.VideoCapture(video_path)
        self.width = int(vcp.get(3))
        self.height = int(vcp.get(4))
        self.frame_rate = int(vcp.get(5))
        self.frame_num = -1
        self.frame_data = []

        ret = True
        while(ret):
            ret, frame = vcp.read()
            if frame is None:
                continue
            if frame_size:
                frame = cv2.resize(frame, frame_size)
            self.frame_data.append(frame)
            self.frame_num += 1
        print('Parsing done.\nvideo: {}\nwidth: {}\nheight: {}\nframe rate: {}\nframe num: {}'.format(
            self.video_basename, self.width, self.height, self.frame_rate, self.frame_num))

    def save_frames(self, save_dir):
        print('Saving the frames of {} to {}.'.format(
            self.video_basename, save_dir))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        pbar = tqdm(total=self.frame_num)
        for i, frame in enumerate(self.frame_data):
            cv2.imwrite(os.path.join(save_dir, '{}_{}.jpg'.format(
                self.video_basename, i)), frame)
            pbar.update()
        pbar.close()
        print('Saving done.')

    def play(self):
        pbar = tqdm(total=self.frame_num)
        for frame in self.frame_data[-10:]:
            cv2.imshow('play', frame)
            cv2.waitKey(self.frame_rate)
            pbar.update()
        pbar.close()
        cv2.waitKey(0)


def compose_dir(input_dir, output_path, start=None, end=None):
    '''
    Video synthesis.
    '''
    files = [os.path.join(input_dir, i) for i in os.listdir(input_dir)]
    files.sort(key=lambda x: int(os.path.splitext(
        os.path.basename(x))[0].split('_')[-1]))
    img = cv2.imread(files[0])
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # 1 file name 2 encoder 3 frame rate 4 frame size
    videoWrite = cv2.VideoWriter(output_path, fourcc, 15, size)
    if not end:
        end = len(files)
    if not start:
        start = 0
    assert start < end
    pbar = tqdm(total=end-start)
    for file in files[start:end] if start or end else files:
        img = cv2.imread(file)
        imgInfo = img.shape
        videoWrite.write(img)
        pbar.update()
    pbar.close()


def compose_data(data, output_path, size, start=None, end=None):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # 1 file name 2 encoder 3 frame rate 4 frame size
    videoWrite = cv2.VideoWriter(output_path, fourcc, 15, size)
    if not end:
        end = len(files)
    if not start:
        start = 0
    assert start < end
    pbar = tqdm(total=end-start)
    for img in data[start:end] if start or end else files:
        videoWrite.write(img)
        pbar.update()
    pbar.close()
