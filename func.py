import os
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


def MotionCorrection(LoadPath: str, SavePath: str, Flag = 1, IfRigid = False):
    """
    Flag: int, default: 1
        Flag == 0: set first frame as template image
        Flag == 1: set average of registered random 20 frames as template image
        Flag == 2: set average of registered first 20 frames as template image  
    IfRigid: bool, default: False
        IfRigid == True: Translation + Rotation (low efficiency)
        IfRigid == False: Translation (recommanded)
    """

    # Load model
    script_path=os.path.abspath(__file__)
    dir_path=os.path.dirname(script_path)
    os.add_dll_directory(dir_path)
    so = ct.cdll.LoadLibrary("./FIFER.dll")

    # Load video and set vars
    InputStack = LoadData(LoadPath)
    OutputStack = np.zeros(InputStack.shape, dtype=np.uint8)
    pInputStack = InputStack.ctypes.data_as(ct.POINTER(ct.c_uint8))
    pOutputStack = OutputStack.ctypes.data_as(ct.POINTER(ct.c_uint8))
    C, Rows, Cols = InputStack.shape

    # Excute motion correction
    so.MotionCorrection(pInputStack, pOutputStack, C, Rows, Cols, Flag, IfRigid)
    
    # save registered video
    fps = 20  # set by yourself
    Write(SavePath, OutputStack, fps)


if __name__ == '__main__':
    MotionCorrection("Test.avi", "Result.avi")
