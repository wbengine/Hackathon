#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import imutils
from threading import Thread


# 多线程，高效读视频
class WebcamVideoStream:
    def __init__(self, src, width=None, height=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.width = width
        self.height = height

        self._read_and_resize()
        print(self.frame.shape)

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def _read_and_resize(self):
        self.grabbed, frame = self.stream.read()
        self.frame = imutils.resize(frame, width=self.width, height=self.height)

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            self._read_and_resize()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    # 使用方法
    video_capture = WebcamVideoStream(src=0, width=600).start()

    while True:
        frame = video_capture.read()
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.stop()
    cv2.destroyAllWindows()
