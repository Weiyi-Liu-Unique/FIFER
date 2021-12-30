import os
import cv2
import numpy as np
from PyQt5 import Qt, QtWidgets
from UI import Mode, MainWidget, Parameter, Parameter1, Parameter2


class BaseFunction:
    def __init__(self):
        # self.NearPts
        # self.h
        # self.HalfSize
        # self.Length
        # self.Range
        pass

    def Step(self, Coordinate: np.ndarray, Weight: np.ndarray) -> (np.ndarray, np.ndarray):
        Distance = (Coordinate[:, None] - self.NearPts) / self.h
        Kernel = Weight * (np.exp((-0.5 * (Distance ** 2).sum(-1))) / (np.pi * 2))
        SumK = Kernel.sum(1)
        Density = SumK / (Weight.sum(1) * self.h * self.h)
        # Density = SumK
        Coordinate_New = ((self.NearPts.T * Kernel[:, None]).sum(-1).T / SumK).T
        return Coordinate_New, Density

    def StepNoDensity(self, Coordinate: np.ndarray, Weight: np.ndarray) -> np.ndarray:
        Distance = (Coordinate[:, None] - self.NearPts) / self.h
        Kernel = Weight * (np.exp((-0.5 * (Distance ** 2).sum(-1))) / (np.pi * 2))
        SumK = Kernel.sum(1)
        Coordinate_New = ((self.NearPts.T * Kernel[:, None]).sum(-1).T / SumK).T
        # print(Coordinate_New)
        return Coordinate_New

    def Density(self, Coordinate: np.ndarray, Weight: np.ndarray) -> np.ndarray:
        Distance = (Coordinate[:, None] - self.NearPts) / self.h
        Kernel = Weight * (np.exp((-0.5 * (Distance ** 2).sum(-1))) / (np.pi * 2))
        Density = Kernel.sum(1) / (Weight.sum(1) * self.h * self.h)
        # Density = Kernel.sum(1)
        return Density

    def GenerateWeight(self, Coordinate: np.ndarray, Img: np.ndarray) -> np.ndarray:
        Weight = np.zeros((Coordinate.shape[0], self.NearPts.shape[0]))
        LeftTop = Coordinate - self.HalfSize
        RightBottom = Coordinate + self.HalfSize + 1
        for i in range(Coordinate.shape[0]):
            Mask = cv2.resize(Img[LeftTop[i, 0]: RightBottom[i, 0], LeftTop[i, 1]: RightBottom[i, 1]],
                              dsize=(self.Length, self.Length))
            Weight[i, :] = Mask.ravel()
        return Weight

    def FilterOutRange(self, Coordinate: np.ndarray) -> np.ndarray:
        Coordinate = Coordinate[
            np.logical_and(np.logical_and(Coordinate[:, 0] > self.Range[0], Coordinate[:, 0] < self.Range[2]),
                           np.logical_and(Coordinate[:, 1] > self.Range[1], Coordinate[:, 1] < self.Range[3]))]
        return Coordinate


class MyDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mouseReleaseEvent(self, MouseEvent):
        super().mouseReleaseEvent(MouseEvent)
        if MouseEvent.button() == Qt.Qt.LeftButton:
            x = int((MouseEvent.x() - 40) * self.Img.shape[1] / 400)
            y = int((MouseEvent.y() - 40) * self.Img.shape[0] / 400)
            try:
                if self.HalfSize < x < self.Img.shape[1] - self.HalfSize and self.HalfSize < y < self.Img.shape[
                    0] - self.HalfSize:
                    self.Kernel_Img = self.Img_BGR[(y - self.HalfSize): (y + self.HalfSize + 1),
                                      (x - self.HalfSize): (x + self.HalfSize + 1), :]
                    self.Kernel_Label1.setGeometry(Qt.QRect(100, 500, self.Size, self.Size))
                    self.Kernel_Label1.setPixmap(Window.Mat2Qpix(self.Kernel_Img))

                    Coordinate = np.arange(-self.r * (self.HalfSize // self.r), self.r * (self.HalfSize // self.r) + 1,
                                           self.r)
                    self.Length = Coordinate.shape[0]
                    y, x = np.meshgrid(Coordinate, Coordinate)
                    self.NearPts = np.array(
                        [x.ravel(), y.ravel()]).T  # Mesh-grid position of nearby points in a standard kernel.

                    self.Kernel_Sampled_Img = cv2.resize(self.Kernel_Img, (self.Length, self.Length))
                    self.Kernel_Label2.setGeometry(Qt.QRect(300, 500, self.Length, self.Length))
                    self.Kernel_Label2.setPixmap(Window.Mat2Qpix(self.Kernel_Sampled_Img))
            except:
                pass


class SetMode(QtWidgets.QDialog, Mode.Ui_Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.setupUi(self, ScaleRate)
        self.setupUi(self)


class SetParameter(QtWidgets.QDialog, Parameter.Ui_Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)


class SetParameter1(MyDialog, Parameter1.Ui_Dialog, BaseFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

    def P1_Slider_Changed(self, Value):
        self.P1_Box.blockSignals(True)
        self.P1_Box.setValue(Value)
        self.P1_Box.blockSignals(False)
        self.Threshold = Value

        # ********** FAST feature detection **********
        FAST_Detector = cv2.FastFeatureDetector().create()
        FAST_Detector.setThreshold(Value)
        Coordinate_FAST = FAST_Detector.detect(self.Img)
        Coordinate_FAST = np.array([[_.pt[1], _.pt[0]] for _ in Coordinate_FAST], dtype=np.int32)
        Img_BGR = self.Img_BGR.copy()
        for i in range(Coordinate_FAST.shape[0]):
            cv2.circle(Img_BGR, tuple(Coordinate_FAST[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)
        self.Img_Label.setPixmap(Window.Mat2Qpix(Img_BGR))
        self.P1_Label2.setText('Number Of FAST Feature Points: ' + str(len(Coordinate_FAST)))

        self.Coordinate_FAST = Coordinate_FAST

        self.P2_Slider.setEnabled(True)
        self.P2_Box.setEnabled(True)
        # ********** FAST feature detection **********

    def P1_Box_Changed(self, Value):
        self.P1_Slider.blockSignals(True)
        self.P1_Slider.setValue(Value)
        self.P1_Slider.blockSignals(False)
        self.Threshold = Value

        # ********** FAST feature detection **********
        FAST_Detector = cv2.FastFeatureDetector().create()
        FAST_Detector.setThreshold(Value)
        Coordinate_FAST = FAST_Detector.detect(self.Img)
        Coordinate_FAST = np.array([[_.pt[1], _.pt[0]] for _ in Coordinate_FAST], dtype=np.int32)
        Img_BGR = self.Img_BGR.copy()
        for i in range(Coordinate_FAST.shape[0]):
            cv2.circle(Img_BGR, tuple(Coordinate_FAST[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)
        self.Img_Label.setPixmap(Window.Mat2Qpix(Img_BGR))
        self.P1_Label2.setText('Number Of FAST Feature Points: ' + str(len(Coordinate_FAST)))

        self.Coordinate_FAST = Coordinate_FAST

        self.P2_Slider.setEnabled(True)
        self.P2_Box.setEnabled(True)
        # ********** FAST feature detection **********

    def P2_Slider_Changed(self, Value):
        self.P2_Box.blockSignals(True)
        self.P2_Box.setValue(Value)
        self.P2_Box.blockSignals(False)
        self.R_Square = Value

        Merged_Coordinate = self.Merge(self.Coordinate_FAST, Value)
        self.Merged_Coordinate = Merged_Coordinate
        Img_BGR = self.Img_BGR.copy()
        for i in range(Merged_Coordinate.shape[0]):
            cv2.circle(Img_BGR, tuple(Merged_Coordinate[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)
        self.Img_Label.setPixmap(Window.Mat2Qpix(Img_BGR))
        self.P2_Label2.setText('Number Of Merged FAST Feature Points: ' + str(Merged_Coordinate.shape[0]))

        self.P3_Box1.setEnabled(True)
        self.P1_Slider.setEnabled(False)
        self.P1_Box.setEnabled(False)

    def P2_Box_Changed(self, Value):
        self.P2_Slider.blockSignals(True)
        self.P2_Slider.setValue(Value)
        self.P2_Slider.blockSignals(False)
        self.R_Square = Value

        Merged_Coordinate = self.Merge(self.Coordinate_FAST, Value)
        self.Merged_Coordinate = Merged_Coordinate
        Img_BGR = self.Img_BGR.copy()
        for i in range(Merged_Coordinate.shape[0]):
            cv2.circle(Img_BGR, tuple(Merged_Coordinate[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)
        self.Img_Label.setPixmap(Window.Mat2Qpix(Img_BGR))
        self.P2_Label2.setText('Number Of Merged FAST Feature Points: ' + str(Merged_Coordinate.shape[0]))

        self.P3_Box1.setEnabled(True)
        self.P1_Slider.setEnabled(False)
        self.P1_Box.setEnabled(False)

    def P3_Slider1_Changed(self, Value):
        pass

    def P3_Box1_Changed(self, Value):
        self.Size = Value
        self.P3_Slider1.setValue(Value)
        self.HalfSize = int((Value - 1) / 2)
        self.Range = [self.HalfSize, self.HalfSize, self.Img.shape[0] - 1 - self.HalfSize,
                      self.Img.shape[1] - 1 - self.HalfSize]
        Max = self.HalfSize - 1
        self.P3_Slider2.setRange(1, Max)
        self.P3_Box2.setRange(1, Max)

        self.P3_Box2.setEnabled(True)
        self.P3_Box3.setEnabled(True)
        self.P2_Slider.setEnabled(False)
        self.P2_Box.setEnabled(False)
        self.Img_Label.setPixmap(Window.Mat2Qpix(self.Img_BGR))
        self.r = self.P3_Slider2.value()
        self.h = self.P3_Slider3.value()

    def P3_Slider2_Changed(self, Value):
        pass

    def P3_Box2_Changed(self, Value):
        self.r = Value
        self.P3_Slider2.setValue(Value)

        self.P3_Slider3.setRange(Value, 2 * Value)
        self.P3_Box3.setRange(Value, 2 * Value)
        self.P3_Slider3.setValue(2 * Value)
        self.P3_Box3.setValue(2 * Value)
        self.P3_Box1.setEnabled(False)

    def P3_Slider3_Changed(self, Value):
        pass

    def P3_Box3_Changed(self, Value):
        self.h = Value
        self.P3_Slider3.setValue(Value)
        self.P3_Box1.setEnabled(False)

    @staticmethod
    def Merge(Coordinate: np.ndarray, r_square: int) -> np.ndarray:
        CombinePts = []
        while True:
            Len = Coordinate.shape[0]
            if Len == 0:
                break
            p = np.random.randint(Len)
            homogeneous = ((Coordinate - Coordinate[p]) ** 2).sum(1) < r_square
            Homo = Coordinate[homogeneous]
            if Homo.shape[0] == 0:
                CombinePts.append([Coordinate[p, 0], Coordinate[p, 1]])
                Coordinate = np.delete(Coordinate, p, axis=0)
            else:
                CombinePts.append([np.mean(Homo[:, 0]), np.mean(Homo[:, 1])])
                Coordinate = Coordinate[homogeneous == 0]
        Coordinate = np.array(CombinePts, dtype=np.int32)
        return Coordinate

    def GenerateInitialPoints(self, Img: np.ndarray) -> np.ndarray:
        self.Epsilon = 0.001
        # ********** FAST feature detection **********
        Coordinate_FAST = self.Merged_Coordinate
        # ********** FAST feature detection **********

        # ********** First Clustering (rough calculation) **********
        Coordinate_FAST = self.FilterOutRange(Coordinate_FAST.astype(int))
        Weight = self.GenerateWeight(Coordinate_FAST, Img)
        Coordinate_Input = np.zeros(Coordinate_FAST.shape)
        Weight_Input = Weight.copy()

        MaxPts = np.zeros(Coordinate_FAST.shape)
        Sequence = np.arange(Coordinate_FAST.shape[0])

        while True:
            Coordinate_New = self.StepNoDensity(Coordinate_Input, Weight_Input)
            IsMaxPts = (np.abs(Coordinate_New - Coordinate_Input) <= 1).sum(1) == 2
            MaxPts[Sequence[IsMaxPts]] = Coordinate_New[IsMaxPts]
            NotMaxPts = (IsMaxPts == 0)
            Coordinate_Input = Coordinate_New[NotMaxPts]
            Weight_Input = Weight_Input[NotMaxPts]
            Sequence = Sequence[NotMaxPts]

            if len(Coordinate_Input) == 0:
                break

        Density_MaxPts = self.Density(MaxPts, Weight)
        Xi = np.percentile(Density_MaxPts, 70)
        NoNoise = Density_MaxPts > Xi
        MaxPts = MaxPts[NoNoise] + Coordinate_FAST[NoNoise]
        Coordinate_First = self.Merge(MaxPts, self.R_Square)
        # ********** First Clustering (rough calculation) **********

        # ********** Second Clustering (precise calculation) **********
        Coordinate_First = self.FilterOutRange(Coordinate_First.astype(int))
        Weight = self.GenerateWeight(Coordinate_First, Img)
        Coordinate_Input = np.zeros(Coordinate_First.shape)
        Weight_Input = Weight.copy()

        MaxPts = np.zeros(Coordinate_First.shape)
        Sequence = np.arange(Coordinate_First.shape[0])

        while True:
            Coordinate_New, Density = self.Step(Coordinate_Input, Weight_Input)
            Density_New = self.Density(Coordinate_New, Weight_Input)
            IsMaxPts = ((Density_New - Density) / Density_New) <= self.Epsilon
            MaxPts[Sequence[IsMaxPts]] = Coordinate_New[IsMaxPts]
            NotMaxPts = (IsMaxPts == 0)
            Coordinate_Input = Coordinate_New[NotMaxPts]
            Weight_Input = Weight_Input[NotMaxPts]
            Sequence = Sequence[NotMaxPts]

            if len(Coordinate_Input) == 0:
                break
        MaxPts += Coordinate_First
        Coordinate_Second = self.Merge(MaxPts, self.R_Square)
        # ********** Second Clustering (precise calculation) **********

        # ********** Third Clustering (to correct errors caused by merging feature points) **********
        Coordinate_Second = self.FilterOutRange(Coordinate_Second.astype(int))
        Weight = self.GenerateWeight(Coordinate_Second, Img)
        Coordinate_Input = np.zeros(Coordinate_Second.shape)
        Weight_Input = Weight.copy()

        MaxPts = np.zeros(Coordinate_Second.shape)
        Sequence = np.arange(Coordinate_Second.shape[0])

        while True:
            Coordinate_New, Density = self.Step(Coordinate_Input, Weight_Input)
            Density_New = self.Density(Coordinate_New, Weight_Input)
            IsMaxPts = ((Density_New - Density) / Density_New) <= self.Epsilon
            MaxPts[Sequence[IsMaxPts]] = Coordinate_New[IsMaxPts]
            NotMaxPts = (IsMaxPts == 0)
            Coordinate_Input = Coordinate_New[NotMaxPts]
            Weight_Input = Weight_Input[NotMaxPts]
            Sequence = Sequence[NotMaxPts]

            if len(Coordinate_Input) == 0:
                break
        MaxPts += Coordinate_Second
        # ********** Third Clustering (to correct errors caused by merging feature points) **********
        return MaxPts


class SetParameter2(QtWidgets.QDialog, Parameter2.Ui_Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.Alpha1 = self.P1_Box1.value()
        self.J = self.P1_Box2.value()

    def P1_Slider1_Changed(self, Value):
        Value = 0.005 * Value
        self.Alpha1 = Value
        self.P1_Box1.setValue(Value)
        self.Match()

    def P1_Box1_Changed(self, Value):
        self.Alpha1 = Value
        self.P1_Slider1.setValue(round(Value / 0.005))
        self.Match()

    def P1_Slider2_Changed(self, Value):
        Value = (Value + 2) / 10
        self.J = Value
        self.P1_Box2.setValue(Value)
        self.Match()

    def P1_Box2_Changed(self, Value):
        self.J = Value
        self.P1_Slider2.setValue(round(Value * 10 - 2))
        self.Match()

    def Match(self):
        def MatchFeaturePoints(Vec1: np.ndarray, Vec2: np.ndarray, Alpha: float, J: float) -> list:
            DescriptorSets1 = np.zeros((Vec1.shape[0], Vec1.shape[0], Vec1.shape[1]))
            DescriptorSets2 = np.zeros((Vec2.shape[0], Vec2.shape[0], Vec2.shape[1]))
            for i in range(Vec1.shape[0]):
                DescriptorSets1[i] = Vec1[i] - Vec1
            for i in range(Vec2.shape[0]):
                DescriptorSets2[i] = Vec2[i] - Vec2

            CoupledPts = []
            for i in range(DescriptorSets1.shape[0]):
                for j in range(DescriptorSets2.shape[0]):
                    Match_Num = 0
                    Descriptor1, Descriptor2 = DescriptorSets1[i], DescriptorSets2[j]
                    Norm_Vec1 = np.sum(Descriptor1 ** 2, axis=1)
                    Norm_Vec2 = np.sum(Descriptor2 ** 2, axis=1)
                    for k in range(Descriptor1.shape[0]):
                        distance = Descriptor1[k] - Descriptor2
                        Distance = np.sum(distance ** 2, axis=1)
                        Norm_Vec = (Norm_Vec1[k] + Norm_Vec2) * 0.5
                        Difference = Distance / Norm_Vec
                        Difference[np.isnan(Difference)] = 1
                        if np.sum(Difference <= Alpha) > 0:
                            Match_Num += 1
                    if Match_Num != 0:
                        J_Similar = Match_Num / (min(Descriptor1.shape[0], Descriptor2.shape[0]) * 2 - Match_Num)
                        if J_Similar > J:
                            CoupledPts.append([i, j])
            return CoupledPts

        Couples = MatchFeaturePoints(self.Coordinate_Tmp_Initial, self.Coordinate_Frame_Initial,
                                     self.Alpha1, self.J)
        Couples = np.array(Couples, dtype=int)

        self.Match_Num = Couples.shape[0]
        self.P1_Label4.setText('Match Number: ' + str(self.Match_Num))

        if self.Match_Num <= 3:
            Plot_Img = np.hstack([self.Tmp, 255 * np.ones((self.Tmp.shape[0], int(self.Tmp.shape[1] / 5))), self.Frame])
            Plot_Img_BGR = cv2.cvtColor(np.uint8(Plot_Img), cv2.COLOR_GRAY2BGR)
            self.Img_Label3.setPixmap(Window.Mat2Qpix(Plot_Img_BGR))
            return

        # To get RANSAC filtered coupled feature points
        Coordinate_Tmp_Coupled = self.Coordinate_Tmp_Initial[Couples[:, 0]]
        Coordinate_Frame_Coupled = self.Coordinate_Frame_Initial[Couples[:, 1]]
        _, Index = cv2.findHomography(Coordinate_Tmp_Coupled[:, [1, 0]], Coordinate_Frame_Coupled[:, [1, 0]],
                                      cv2.RANSAC, 5.0)
        Couples = Couples[Index.ravel().astype(bool)]

        self.Coordinate_Tmp_Coupled = self.Coordinate_Tmp_Initial[Couples[:, 0]]
        self.Coordinate_Frame_Coupled = self.Coordinate_Frame_Initial[Couples[:, 1]]

        Coordinate1_Convert = self.Coordinate_Tmp_Coupled[:, [1, 0]]
        Coordinate2_Convert = self.Coordinate_Frame_Coupled[:, [1, 0]]
        Coordinate2_Convert[:, 0] += self.Tmp.shape[0] + int(self.Tmp.shape[1] / 5)

        Plot_Img = np.hstack([self.Tmp, 255 * np.ones((self.Tmp.shape[0], int(self.Tmp.shape[1] / 5))), self.Frame])

        Plot_Img_BGR = cv2.cvtColor(np.uint8(Plot_Img), cv2.COLOR_GRAY2BGR)

        for i in range(Coordinate1_Convert.shape[0]):
            cv2.circle(Plot_Img_BGR, tuple(Coordinate1_Convert[i]), 5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(Plot_Img_BGR, tuple(Coordinate2_Convert[i]), 5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(Plot_Img_BGR, tuple(Coordinate1_Convert[i]), tuple(Coordinate2_Convert[i]), (0, 0, 255), 2, cv2.LINE_AA)

        self.Img_Label3.setPixmap(Window.Mat2Qpix(Plot_Img_BGR))


class SetParameter2_(QtWidgets.QDialog, Parameter2.Ui_Dialog, BaseFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.Alpha2 = self.P1_Box1.value()

        self.P1_Slider2.setEnabled(False)
        self.P1_Box2.setEnabled(False)
        self.setWindowTitle('Matching Renewed Feature Points')
        self.Label1.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Template (Fixed)</span></p></body></html>")
        self.Label2.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Frame (Renewed)</span></p></body></html>")
        self.Label3.setText(
            "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Matching Feature Points (No Filtration)</span></p></body></html>")
        self.P1_Label2.setText('Alpha2')
        self.P1_Label5.setText("<html><head/><body><p>Recommended Match Number: &gt; 0</p></body></html>")

    def P1_Slider1_Changed(self, Value):
        Value = 0.005 * Value
        self.Alpha2 = Value
        self.P1_Box1.setValue(Value)
        self.Match()

    def P1_Box1_Changed(self, Value):
        self.Alpha2 = Value
        self.P1_Slider1.setValue(round(Value / 0.005))
        self.Match()

    def P1_Slider2_Changed(self, Value):
        pass

    def P1_Box2_Changed(self, Value):
        pass

    def FindLocalMaximum(self, Coordinate: np.ndarray, Img: np.ndarray) -> np.ndarray:
        Img = cv2.GaussianBlur(Img, (5, 5), 0)
        Coordinate = self.FilterOutRange(Coordinate.astype(int))
        Weight = self.GenerateWeight(Coordinate, Img)
        Coordinate_Input = np.zeros(Coordinate.shape)
        Weight_Input = Weight.copy()

        MaxPts = np.zeros(Coordinate.shape)
        Sequence = np.arange(Coordinate.shape[0])

        while True:
            Coordinate_New = self.StepNoDensity(Coordinate_Input, Weight_Input)
            IsMaxPts = (np.abs(Coordinate_New - Coordinate_Input) <= 1).sum(1) == 2
            MaxPts[Sequence[IsMaxPts]] = Coordinate_New[IsMaxPts]

            NotMaxPts = (IsMaxPts == 0)

            Coordinate_Input = Coordinate_New[NotMaxPts]
            Weight_Input = Weight_Input[NotMaxPts]
            Sequence = Sequence[NotMaxPts]

            if len(Coordinate_Input) == 0:
                break

        MaxPts += Coordinate
        return MaxPts

    def Match(self):
        def MatchFeaturePoints(Vec1: np.ndarray, Vec2: np.ndarray, Alpha: float, J: float) -> list:
            DescriptorSets1 = np.zeros((Vec1.shape[0], Vec1.shape[0], Vec1.shape[1]))
            DescriptorSets2 = np.zeros((Vec2.shape[0], Vec2.shape[0], Vec2.shape[1]))
            for i in range(Vec1.shape[0]):
                DescriptorSets1[i] = Vec1[i] - Vec1
            for i in range(Vec2.shape[0]):
                DescriptorSets2[i] = Vec2[i] - Vec2

            CoupledPts = []
            for i in range(DescriptorSets1.shape[0]):
                for j in range(DescriptorSets2.shape[0]):
                    Match_Num = 0
                    Descriptor1, Descriptor2 = DescriptorSets1[i], DescriptorSets2[j]
                    Norm_Vec1 = np.sum(Descriptor1 ** 2, axis=1)
                    Norm_Vec2 = np.sum(Descriptor2 ** 2, axis=1)
                    for k in range(Descriptor1.shape[0]):
                        distance = Descriptor1[k] - Descriptor2
                        Distance = np.sum(distance ** 2, axis=1)
                        Norm_Vec = (Norm_Vec1[k] + Norm_Vec2) * 0.5
                        Difference = Distance / Norm_Vec
                        Difference[np.isnan(Difference)] = 1
                        if np.sum(Difference <= Alpha) > 0:
                            Match_Num += 1
                    if Match_Num != 0:
                        J_Similar = Match_Num / (min(Descriptor1.shape[0], Descriptor2.shape[0]) * 2 - Match_Num)
                        if J_Similar > J:
                            CoupledPts.append([i, j])
            return CoupledPts
        Couples = MatchFeaturePoints(self.Coordinate_Tmp, self.Coordinate_Frame, self.Alpha2, self.J)
        Couples = np.array(Couples, dtype=int)

        self.Match_Num = Couples.shape[0]
        self.P1_Label4.setText('Match Number: ' + str(self.Match_Num))

        if self.Match_Num == 0:
            Plot_Img = np.hstack([self.Tmp, 255 * np.ones((self.Tmp.shape[0], int(self.Tmp.shape[1] / 5))), self.Frame])
            Plot_Img_BGR = cv2.cvtColor(np.uint8(Plot_Img), cv2.COLOR_GRAY2BGR)
            self.Img_Label3.setPixmap(Window.Mat2Qpix(Plot_Img_BGR))
            return

        self.Coordinate_Tmp_Coupled = self.Coordinate_Tmp[Couples[:, 0]]
        self.Coordinate_Frame_Coupled = self.Coordinate_Frame[Couples[:, 1]]

        Coordinate1_Convert = self.Coordinate_Tmp_Coupled[:, [1, 0]]
        Coordinate2_Convert = self.Coordinate_Frame_Coupled[:, [1, 0]]
        Coordinate2_Convert[:, 0] += self.Tmp.shape[0] + int(self.Tmp.shape[1] / 5)

        Plot_Img = np.hstack([self.Tmp, 255 * np.ones((self.Tmp.shape[0], int(self.Tmp.shape[1] / 5))), self.Frame])

        Plot_Img_BGR = cv2.cvtColor(np.uint8(Plot_Img), cv2.COLOR_GRAY2BGR)

        for i in range(Coordinate1_Convert.shape[0]):
            cv2.circle(Plot_Img_BGR, tuple(Coordinate1_Convert[i]), 5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(Plot_Img_BGR, tuple(Coordinate2_Convert[i]), 5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(Plot_Img_BGR, tuple(Coordinate1_Convert[i]), tuple(Coordinate2_Convert[i]), (0, 0, 255), 2, cv2.LINE_AA)

        self.Img_Label3.setPixmap(Window.Mat2Qpix(Plot_Img_BGR))


class Window(QtWidgets.QWidget, MainWidget.Ui_Form):
    def __init__(self):
        super().__init__()
        # self.setupUi(self, ScaleRate)
        self.setupUi(self)
        self.IsGetTmp = False
        self.SrcPath = None

    def BeginProcedure(self):
        if self.SrcPath is None:
            return None
        mode = SetMode(self)
        mode.show()

        def ApplyParameter1():
            if not mode.Mode1.isChecked() and not mode.Mode2.isChecked():
                return None

            P1 = SetParameter1(self)

            if mode.Mode1.isChecked():
                P1.Img = cv2.GaussianBlur(self.Capture_Mat[0], (5, 5), 0)
            else:
                P1.Img = cv2.pyrDown(self.Capture_Mat[0])
            P1.Img_BGR = cv2.cvtColor(P1.Img, cv2.COLOR_GRAY2BGR)
            P1.show()

            def ApplyParameter2():
                print('P2')
                if mode.Mode1.isChecked():
                    Tmp = cv2.GaussianBlur(self.Capture_Mat[0], (5, 5), 0)
                    Frame = cv2.GaussianBlur(self.Capture_Mat[1], (5, 5), 0)
                else:
                    Tmp = cv2.pyrDown(self.Capture_Mat[0])
                    Frame = cv2.pyrDown(self.Capture_Mat[1])
                Coordinate_Tmp_Initial = P1.GenerateInitialPoints(Tmp)
                Coordinate_Frame_Initial = P1.GenerateInitialPoints(Frame)
                Coordinate_Tmp_Initial = Coordinate_Tmp_Initial.astype(int)
                Coordinate_Frame_Initial = Coordinate_Frame_Initial.astype(int)

                Tmp_BGR = cv2.cvtColor(Tmp, cv2.COLOR_GRAY2BGR)
                Frame_BGR = cv2.cvtColor(Frame, cv2.COLOR_GRAY2BGR)

                for i in range(Coordinate_Tmp_Initial.shape[0]):
                    cv2.circle(Tmp_BGR, tuple(Coordinate_Tmp_Initial[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)
                for i in range(Coordinate_Frame_Initial.shape[0]):
                    cv2.circle(Frame_BGR, tuple(Coordinate_Frame_Initial[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)

                P2 = SetParameter2(self)
                P2.Img_Label1.setPixmap(Window.Mat2Qpix(Tmp_BGR))
                P2.Img_Label2.setPixmap(Window.Mat2Qpix(Frame_BGR))

                P2.Tmp = Tmp
                P2.Frame = Frame
                P2.Coordinate_Tmp_Initial = Coordinate_Tmp_Initial
                P2.Coordinate_Frame_Initial = Coordinate_Frame_Initial

                P2.show()

                def ApplyParameter2_():
                    print('P2_')

                    P2_ = SetParameter2_(self)
                    P2_.J = P2.J
                    P2_.P1_Slider2.setValue(int((P2_.J * 10) - 2))
                    P2_.P1_Box2.setValue(P2_.J)

                    P2_.NearPts = P1.NearPts
                    P2_.h = P1.h
                    P2_.HalfSize = P1.HalfSize
                    P2_.Length = P1.Length
                    P2_.Range = P1.Range
                    P2_.Coordinate_Tmp = P2.Coordinate_Tmp_Coupled
                    P2_.Coordinate_Frame = P2.Coordinate_Frame_Coupled

                    # To filter out range feature points.
                    Index = np.logical_and(np.logical_and(P2_.Coordinate_Frame[:, 0] > P2_.Range[0],
                                                          P2_.Coordinate_Frame[:, 0] < P2_.Range[2]),
                                           np.logical_and(P2_.Coordinate_Frame[:, 1] > P2_.Range[1],
                                                          P2_.Coordinate_Frame[:, 1] < P2_.Range[3]))
                    P2_.Coordinate_Tmp = P2_.Coordinate_Tmp[Index]
                    P2_.Coordinate_Frame = P2_.Coordinate_Frame[Index]

                    if mode.Mode1.isChecked():
                        P2_.Tmp = cv2.GaussianBlur(self.Capture_Mat[0], (5, 5), 0)
                        P2_.Frame = cv2.GaussianBlur(self.Capture_Mat[2], (5, 5), 0)
                    else:
                        P2_.Tmp = cv2.pyrDown(self.Capture_Mat[0])
                        P2_.Frame = cv2.pyrDown(self.Capture_Mat[2])

                    P2_.Coordinate_Frame = P2_.FindLocalMaximum(P2_.Coordinate_Frame, P2_.Frame)

                    P2_.Coordinate_Tmp = P2_.Coordinate_Tmp.astype(int)
                    P2_.Coordinate_Frame = P2_.Coordinate_Frame.astype(int)

                    P2_.Tmp_BGR = cv2.cvtColor(Tmp, cv2.COLOR_GRAY2BGR)
                    P2_.Frame_BGR = cv2.cvtColor(Frame, cv2.COLOR_GRAY2BGR)

                    for i in range(P2_.Coordinate_Tmp.shape[0]):
                        cv2.circle(P2_.Tmp_BGR, tuple(P2_.Coordinate_Tmp[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)
                    for i in range(P2_.Coordinate_Frame.shape[0]):
                        cv2.circle(P2_.Frame_BGR, tuple(P2_.Coordinate_Frame[i, [1, 0]]), 5, (0, 0, 255), 2, cv2.LINE_AA)

                    P2_.Img_Label1.setPixmap(Window.Mat2Qpix(P2_.Tmp_BGR))
                    P2_.Img_Label2.setPixmap(Window.Mat2Qpix(P2_.Frame_BGR))

                    P2_.show()

                    def ApplyParameter():
                        parameter = [P1.Size, P1.r, P1.h, P1.Threshold, P1.R_Square, P2.Alpha1, P2_.Alpha2, P2_.J]
                        P = SetParameter(self)
                        for i in range(len(parameter)):
                            item = QtWidgets.QTableWidgetItem(str(parameter[i]))
                            P.Parameter.setItem(i - 1, 1, item)
                        P.show()

                        parameter += [self.SrcPath.replace('/', '\\\\'), '.\\\\FIFER_Output.avi']
                        # print(parameter)

                        para_name = ['#Size=\n', '#r=\n', '#h=\n', '#Threshold=\n', '#R_Square=\n', '#Alpha1=\n',
                                     '#Alpha2=\n', '#J=\n', '#LoadPosition=\n', '#WritePosition=\n']

                        Lines = ''
                        for i in range(len(parameter)):
                            if type(parameter[i]) != str:
                                Lines += para_name[i] + str(parameter[i]) + '\n'
                            else:
                                Lines += para_name[i] + parameter[i] + '\n'

                        WriteParameter = open(".\\Parameter.txt", 'w+', encoding='utf-8')
                        WriteParameter.write(Lines)
                        WriteParameter.close()

                        if mode.Mode1.isChecked():
                            path = os.getcwd() + '\\Core\\Normal.exe'
                            os.system(path)
                        else:
                            path = os.getcwd() + '\\Core\\DownScale.exe'
                            os.system(path)

                    P2_.buttonBox.accepted.connect(ApplyParameter)

                P2.buttonBox.accepted.connect(ApplyParameter2_)

            P1.buttonBox.accepted.connect(ApplyParameter2)

        mode.buttonBox.accepted.connect(ApplyParameter1)

    def GetTemplate(self):
        try:
            path = Qt.QFileDialog.getOpenFileNames(self, 'Select', './', 'Images(*.jpg *.png *.jpeg)')[0][0]
            TmpPix = Qt.QPixmap(path)
            Info = "<html><head/><body><p>Location: " + path + "</p><p>Size: " + str(TmpPix.width()) + "x" + str(
                TmpPix.height()) + "</p></body></html>"
            self.Tmp.setPixmap(TmpPix)
            self.TmpInfo.setText(Info)
            self.IsGetTmp = True
        except:
            pass

    def GetSource(self):
        try:
            path = Qt.QFileDialog.getOpenFileNames(self, 'Select', './', 'Video(*.avi)')[0][0]
            cap = cv2.VideoCapture(path)
            Capture = []
            self.Capture_Mat = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.Capture_Mat.append(frame)
                    Capture.append(self.Mat2Qpix(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)))
                else:
                    break
            self.SrcImg = Capture
            SrcPix = Capture[0]
            Info = "<html><head/><body><p>Location: " + path + "</p><p>Size: " + str(SrcPix.width()) + "x" + str(
                SrcPix.height()) + "</p><p>Frame: " + str(len(self.SrcImg)) + "</p></body></html>"
            self.Src.setPixmap(SrcPix)
            self.SrcInfo.setText(Info)
            self.SrcPath = path

            if not self.IsGetTmp:
                TmpPix = SrcPix
                InfoTmp = "<html><head/><body><p>Location: First Frame</p><p>Size: " + str(TmpPix.width()) + "x" + str(
                    TmpPix.height()) + "</p></body></html>"
                self.Tmp.setPixmap(TmpPix)
                self.TmpInfo.setText(InfoTmp)
                self.IsGetTmp = True
        except:
            pass

    def ChangeTmpPix(self):
        self.TmpSlider.setValue(0)

    def ChangeSrcPix(self, val):
        try:
            if len(self.SrcImg) > 1:
                self.SrcSlider.setRange(1, len(self.SrcImg))
                self.Src.setPixmap(self.SrcImg[val - 1])
            else:
                self.SrcSlider.setValue(0)
        except:
            self.SrcSlider.setValue(0)

    @staticmethod
    def Mat2Qpix(img):
        # BGR
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Qt.QPixmap.fromImage(Qt.QImage(img.data, width, height, bytesPerLine, Qt.QImage.Format_RGB888))


if __name__ == '__main__':
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    app = QtWidgets.QApplication(sys.argv)
    # app.addLibraryPath('../Qt/plugins')

    window = Window()
    window.show()

    sys.exit(app.exec_())
