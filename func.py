import time
import cv2
import numpy as np
import ctypes as ct

np.set_printoptions(suppress=True)


def Write(path, frames, fps=20):
    size = frames[0].shape
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size, isColor=False)
    for frame in frames:
        video.write(frame)
    video.release()


def LoadData(path: str):
    ImgSet = []
    Capture = cv2.VideoCapture(path)
    while True:
        ret, Frame = Capture.read()
        if ret:
            Frame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
            ImgSet.append(Frame)
        else:
            break
    ImgSet = np.array(ImgSet)
    return ImgSet


def MotionCorrection(LoadPath, SavePath=None):
    so = ct.cdll.LoadLibrary("FIFER_LIB.dll")
    InputStack = LoadData(LoadPath)
    OutputStack = np.zeros(InputStack.shape, dtype=np.uint8)
    pInputStack = InputStack.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pOutputStack = OutputStack.ctypes.data_as(ct.POINTER(ct.c_uint8))
    C, Rows, Cols = InputStack.shape
    IfRigid = False
    so.MotionCorrection(pInputStack, pOutputStack, C, Rows, Cols, IfRigid)
    if SavePath is not None:
        Write(SavePath, OutputStack, 20)


if __name__ == '__main__':
    MotionCorrection("Test.avi", "Result.avi")
