import os
import cv2
import numpy as np
from until import until
import shutil
class Crop():
    def __init__(self, datasetRoot, cropRoot, paddingRadio, imgExe='.png',showRect=False, cropSizes=None, cropClasses=None, opts=None):
        '''
        从数据集中裁剪目标，目前按照目标范围的2倍裁剪
        :param datasetRoot: 数据集根路径，数据集的格式使用jing自己的格式，即[info, imgW, imgH, objClass, xmin, ymin, xmax, ymax]
        :param cropRoot: 裁剪结果输出根目录，根目录下有images/[train, test]和labels/[train, test]
        :param paddingRadio: padding是原始目标尺寸的比例范围
        :param showRect: 是否在裁剪的输出图片中显示目标框
        :param cropSizes: 裁剪的目标尺寸要求
        :param cropClasses: 裁剪的目标类型要求
        :param opts: 对训练集还是测试集进行操作
        '''
        if opts is None:
            opts = ['train']
        self.datasetRoot = datasetRoot
        self.cropRoot = cropRoot
        self.paddingRadio = paddingRadio
        self.imgExe = imgExe
        if (os.path.exists(self.cropRoot)):
            shutil.rmtree(self.cropRoot)
        os.makedirs(self.cropRoot + '/images')
        os.makedirs(self.cropRoot + '/labels')
        self.showRect = showRect
        self.cropSizes = cropSizes
        self.cropClasses = cropClasses
        self.opts = opts
        self.curOpt = opts[0]  # 当前处理的数据集是训练集还是测试集

        self.info = None  # 记录当前正在处理裁剪的图片的info
        self.curImg = None  # 记录当前正在处理裁剪的图片
        self.curVisited = None  # 记录当前正在处理裁剪的图片的labels是否被裁剪过
        self.curLabels = None  # 记录当前正在处理裁剪的图片的labels
    def tryCrop(self, labelIdx):
        label = self.curLabels[labelIdx]
        cropIdxs = [labelIdx]
        objW = label[6] - label[4]  # 标记裁剪之前的多个目标区域的宽
        objH = label[7] - label[5]  # 标记裁剪之前的多个目标区域的高
        randomPaddingRadio = np.random.uniform(self.paddingRadio[0], self.paddingRadio[1])
        cropW = int(objW * randomPaddingRadio)
        cropH = int(objH * randomPaddingRadio)
        cropXmin = max(0, label[4] - cropW // 2)  # 标记此次尝试裁剪区域
        cropYmin = max(0, label[5] - cropH // 2)
        cropXmax = min(label[1], label[6] + cropW // 2)
        cropYmax = min(label[2], label[7] + cropH // 2)
        needTry = True  # 是否需要继续裁剪尝试
        while(needTry):
            isBreak = False
            for i in range(len(self.curLabels)):
                if(self.curVisited[i] == 1):
                    continue
                label = self.curLabels[i]  # 某一个目标的label
                xmin = min(cropXmin, label[4])
                ymin = min(cropYmin, label[5])
                xmax = max(cropXmax, label[6])
                ymax = max(cropYmax, label[7])
                if(xmax - xmin < cropXmax - cropXmin + label[6] - label[4] and ymax - ymin < cropYmax - cropYmin + label[7] - label[5]):  # 目标和尝试裁剪的区域有重叠
                    cropIdxs.append(i)
                    self.curVisited[i] = 1
                    objsXmin = label[1]  # 标记此次裁剪区域中的目标区域
                    objsYmin = label[2]
                    objsXmax = objsYmax = 0
                    for cropIdx in cropIdxs:
                        label = self.curLabels[cropIdx]
                        objsXmin = min(objsXmin, label[4])
                        objsYmin = min(objsYmin, label[5])
                        objsXmax = max(objsXmax, label[6])
                        objsYmax = max(objsYmax, label[7])
                    objW = objsXmax - objsXmin
                    objH = objsYmax - objsYmin
                    randomPaddingRadio = np.random.uniform(self.paddingRadio[0], self.paddingRadio[1])
                    cropW = int(objW * randomPaddingRadio)
                    cropH = int(objH * randomPaddingRadio)
                    cropXmin = max(0, objsXmin - cropW // 2)
                    cropYmin = max(0, objsYmin - cropH // 2)
                    cropXmax = min(label[1], objsXmax + cropW // 2)
                    cropYmax = min(label[2], objsYmax + cropH // 2)
                    isBreak = True  # 标记是否for循环直接break出来
                    break
            if(isBreak == False):
                needTry = False
        if not (cropXmax - cropXmin) % 2 == 0:
            cropXmin += 1
        if not (cropYmax - cropYmin) % 2 == 0:
            cropYmin += 1
        trySuccess = True
        if(self.cropClasses != None):
            for cropIdx in cropIdxs:  # 检查裁剪类型
                objClass = int(self.curLabels[cropIdx][3])
                if not objClass in self.cropClasses:
                    trySuccess = False
                    break
        if (self.cropSizes and trySuccess):
            for cropIdx in cropIdxs:  # 检查裁剪尺寸
                label = self.curLabels[cropIdx]
                objW = label[6] - label[4]  # 标记裁剪之前的目标区域的宽
                objH = label[7] - label[5]  # 标记裁剪之前的目标区域的高
                objWH = objW * objH
                hasSize = False
                for cropSize in self.cropSizes:
                    if(cropSize[0] * cropSize[0] < objWH and objWH <= cropSize[1] * cropSize[1]):
                        hasSize = True
                        break
                if not hasSize:
                    trySuccess = False
        if(trySuccess):
            return cropIdxs, [cropXmin, cropYmin, cropXmax, cropYmax]
        else:
            for cropIdx in cropIdxs:
                self.curVisited[cropIdx] = 0
            return [], None
    def writeImgAndLabels(self, cropIdxs, cropArea, cropInfo):
        imgCrop = self.curImg[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]
        fOut = open(self.cropRoot + '/labels/' + self.curOpt + '/' + cropInfo + '.txt', 'w')
        for cropIdx in cropIdxs:
            label = self.curLabels[cropIdx]
            cropImgW = cropArea[2] - cropArea[0]
            cropImgH = cropArea[3] - cropArea[1]
            objClass = label[3]
            xmin = label[4] - cropArea[0]
            ymin = label[5] - cropArea[1]
            xmax = label[6] - cropArea[0]
            ymax = label[7] - cropArea[1]
            if(self.showRect):
                cv2.rectangle(imgCrop, (xmin, ymin), (xmax, ymax), (255,0,0))
            fOut.write(cropInfo + ' ' + str(cropImgW) + ' ' + str(cropImgH) + ' ' + objClass + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')
        cv2.imwrite(self.cropRoot + '/images/' + self.curOpt + '/' + cropInfo + self.imgExe, imgCrop)
    def cropObject(self):
        for opt in self.opts:
            self.curOpt = opt
            os.makedirs(self.cropRoot + '/images/' + opt)
            os.makedirs(self.cropRoot + '/labels/' + opt)
            files = os.listdir(self.datasetRoot + '/labels/' + opt + '/')
            for file in files:
                self.info = file.strip().split('.')[0]
                print(self.info)
                self.curImg = cv2.imread(self.datasetRoot + '/images/' + opt + '/' + self.info + self.imgExe)
                imgH = self.curImg.shape[0]
                imgW = self.curImg.shape[1]
                labels = open(self.datasetRoot + '/labels/' + opt + '/' + self.info + '.txt').readlines()
                self.curLabels = []
                for i in range(len(labels)):
                    label = labels[i]
                    items = label.strip().split(' ')
                    [objClass, xmin, ymin, xmax, ymax] = [items[3], int(items[4]), int(items[5]), int(items[6]), int(items[7])]
                    self.curLabels.append([self.info, imgW, imgH, objClass, xmin, ymin, xmax, ymax])
                self.curVisited = np.zeros(len(self.curLabels))
                cropImgIdx = 0
                for i in range(len(self.curLabels)):
                    if(self.curVisited[i] == 1):
                        continue
                    self.curVisited[i] = 1
                    cropIdxs, cropArea = self.tryCrop(i)
                    if(cropArea != None):
                        self.writeImgAndLabels(cropIdxs, cropArea, self.info + '_' + str(cropImgIdx))
                        cropImgIdx += 1
if __name__ == '__main__':
    datasetRoot = './dataTest_jing/'
    cropRoot = './dataTest_jing/objCrop/padding/1'
    paddingRadio = [1, 1] # paddingRadio参数，这边设置了一个随机范围
    crop = Crop(datasetRoot, cropRoot, paddingRadio, '.png')
    crop.cropObject()