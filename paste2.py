import os
import cv2
import numpy as np
from until import until
import shutil
def createCropIdxNumsArr(boxNum, penNum):
    '''
    有boxNum个箱子，要把penNum支笔均匀放入
    :param imgNum:
    :param cropImgPasteNum:
    :return:
    '''
    baseNum = penNum // boxNum
    emptySeat = penNum % boxNum

    arr1 = np.full(boxNum - emptySeat, baseNum, dtype=int)
    arr2 = np.full(emptySeat, baseNum + 1, dtype=int)
    return np.append(arr1, arr2)
class Paste():
    def __init__(self, datasetRoot, cropRoot, outputRoot, imgNum, pasteRadio=1.0, imgExe='.png',
                 imgSize=None, cropSizes=None, cropClasses=None, showRect=False, randomResizeRatioRange=None, opts=None):
        if imgSize is None:
            imgSize = [416, 416]
        if randomResizeRatioRange is None:
            randomResizeRatioRange = [1, 1]
        if opts is None:
            opts = ['train']
        self.datasetRoot = datasetRoot
        self.cropRoot = cropRoot
        self.outputRoot = outputRoot
        if (os.path.exists(self.outputRoot)):
            shutil.rmtree(self.outputRoot)
        os.makedirs(self.outputRoot + '/images')
        os.makedirs(self.outputRoot + '/labels')
        self.imgNum = imgNum

        self.pasteRadio = pasteRadio  # 实际粘贴的总目标数为：裁剪出来的图片数*pasteRadio
        self.imgExe = imgExe
        self.imgSize = imgSize
        self.cropSizes = cropSizes
        self.cropClasses = cropClasses

        self.showRect = showRect
        self.randomResizeRatioRange = randomResizeRatioRange
        self.opts = opts
        self.curOpt = opts[0]


    def pasteCropsOnImg(self, img, labels, cropImgs, CropImgsLabels, info):
        '''
        尝试把cropImgs粘贴到原始图中
        :param img:
        :param labels:
        :param cropImgs:
        :param CropImgsLabels:
        :return:
        '''
        imgW = img.shape[1]
        imgH = img.shape[0]
        for i in range(len(cropImgs)):
            cropImg = cropImgs[i]
            cropImgLabels = CropImgsLabels[i]
            allowPatse = True
            for cropImgLabel in cropImgLabels:
                radio = min(self.imgSize[0] / imgW, self.imgSize[1] / imgH)
                labelW = (int(cropImgLabel[6]) - int(cropImgLabel[4])) * radio
                labelH = (int(cropImgLabel[7]) - int(cropImgLabel[5])) * radio
                labelWH = int(labelW * labelH)
                incropSizes = False
                for cropSize in self.cropSizes: # 尺寸过滤得在实际知道目标要粘贴到哪张图中才能知道要不要过滤。因为同样的原始目标，在不同图缩小到416时尺寸不同
                    if (cropSize[0] * cropSize[0] <= labelWH and labelWH <= cropSize[1] * cropSize[1]):
                        incropSizes = True
                        break
                if (incropSizes == False):
                    allowPatse = False
                    break
            if(allowPatse == False):  # 一张cropImg中有一个标签不符合尺寸过滤器，就不能粘贴
                continue
            cropImgW = cropImg.shape[1]
            cropImgH = cropImg.shape[0]
            trying = True
            tryNum = 10
            [pasteXmin, pasteYmin, pasteXmax, pasteYmax] = [0, 0 ,0 ,0]
            while(trying and tryNum > 0):
                tryNum -= 1
                [ramdomXmin, ramdomXmax, ramdomYmin, ramdomYmax] = [cropImgW // 2 + 1, imgW - cropImgW // 2 -1, cropImgH // 2 + 1, imgH - cropImgH // 2 -1]
                if(ramdomXmin >= ramdomXmax or ramdomYmin >= ramdomYmax):
                    break
                [xc, yc] = [np.random.randint(ramdomXmin, ramdomXmax), np.random.randint(ramdomYmin, ramdomYmax)]
                overlap = False
                pasteXmin = xc - cropImgW // 2
                pasteYmin = yc - cropImgH // 2
                pasteXmax = xc + cropImgW // 2
                pasteYmax = yc + cropImgH // 2
                for label in labels:
                    if(until.mat_inter( label[4:8], [pasteXmin, pasteYmin, pasteXmax, pasteYmax] ) ):
                        overlap = True
                        break
                if(overlap == False):
                    trying = False
            if(trying == True): # 尝试粘贴失败，直接返回原图即标注
                print("paste fail")
                continue
            img[pasteYmin:pasteYmax, pasteXmin:pasteXmax] = cropImg
            for cropImgLabel in cropImgLabels:
                appendArr = [info, imgW, imgH, cropImgLabel[3], pasteXmin + cropImgLabel[4], pasteYmin + cropImgLabel[5], pasteXmin + cropImgLabel[6], pasteYmin + cropImgLabel[7]]
                labels.append(appendArr)
        return img, labels

    def createCropImgsAndImgsLabels(self, cropIdxNum):
        '''
        寻找cropIdxNum张裁剪图，用于后续粘贴到某一张图片上
        :param cropIdxNum:
        :return: newCropImgs = [cropImg1, cropImg2, ...], newCropImgsLabels = [[label1, label2, ...], [label1, label2, ...], ...]
        '''
        cropImgsLen = len(self.files)
        newCropImgsLabels = []
        newCropImgs = []
        resIdx = 0
        for pasteCount in range(cropIdxNum):
            pasteCropInfo = np.random.randint(0, cropImgsLen)  # 随机找另一种裁剪图
            file = self.files[pasteCropInfo]
            newCropImgsLabels.append([])
            cropInfo = file.strip().split('.')[0]
            cropImg = cv2.imread(
                self.cropRoot + '/images/' + self.curOpt + '/' + cropInfo + self.imgExe)  # 需要粘贴的裁剪图片
            cropImgH = cropImg.shape[0]
            cropImgW = cropImg.shape[1]
            cropLabelsStr = open(self.cropRoot + '/labels/' + self.curOpt + '/' + cropInfo + '.txt',
                                 'r').readlines()

            randomResizeRatio = np.random.uniform(self.randomResizeRatioRange[0],
                                                  self.randomResizeRatioRange[1])
            newCropImgH = int(cropImgH * randomResizeRatio) // 2 * 2
            newCropImgW = int(cropImgW * randomResizeRatio) // 2 * 2
            newCropImgWH = newCropImgW * newCropImgH
            # allowPatse = False
            # for cropSize in self.cropSizes:
            #     if(cropSize[0]*cropSize[0] <= newCropImgWH and newCropImgWH <= cropSize[1]*cropSize[1]):
            #         allowPatse = True
            #         break
            # if(allowPatse == False):
            #     continue
            newCropImg = cv2.resize(cropImg, (newCropImgW, newCropImgH))
            newCropImgs.append(newCropImg)
            hasObjNotInObjsizes = False
            for i in range(len(cropLabelsStr)):
                label = cropLabelsStr[i]
                items = label.strip().split(' ')
                objClass = items[3]
                if(int(objClass) not in self.cropClasses):
                    hasObjNotInObjsizes = True
                xmin = int(float(items[4]) * randomResizeRatio)
                ymin = int(float(items[5]) * randomResizeRatio)
                xmax = int(float(items[6]) * randomResizeRatio)
                ymax = int(float(items[7]) * randomResizeRatio)
                newCropImgsLabels[resIdx].append(
                    [cropInfo, newCropImgW, newCropImgH, objClass, xmin, ymin, xmax, ymax])
                if(hasObjNotInObjsizes):
                    break
            if(hasObjNotInObjsizes == True):
                newCropImgs.pop()
                newCropImgsLabels.pop()
            else:
                resIdx += 1
        return newCropImgs, newCropImgsLabels
    def randomPaste(self):
        for opt in self.opts:
            self.curOpt = opt
            os.makedirs(self.outputRoot + '/images/' + self.curOpt + '/')
            os.makedirs(self.outputRoot + '/labels/' + self.curOpt + '/')
            self.files = os.listdir(self.cropRoot + '/labels/' + self.curOpt + '/')
            cropImgsLen = len(self.files)  # 裁剪出来的图片的数目
            cropImgPasteNum = int(cropImgsLen * self.pasteRadio)
            cropIdxNumsArr = createCropIdxNumsArr(self.imgNum, cropImgPasteNum)
            for idx in range(self.imgNum):
                print(idx)
                info = str(int(idx) % self.imgNum).zfill(5)
                # 处理粘贴图片信息
                cropIdxNum = cropIdxNumsArr[idx]
                newCropImgs, newCropImgsLabels = self.createCropImgsAndImgsLabels(cropIdxNum)
                if(len(newCropImgs) == 0): # 如果没有可行的粘贴目标，直接把原图复制到结果目录
                    shutil.copyfile(self.datasetRoot + '/images/' + self.curOpt + '/' + info + self.imgExe,
                                    self.outputRoot + '/images/' + self.curOpt + '/' + info + self.imgExe)
                    if os.path.exists(self.datasetRoot + '/labels/' + self.curOpt + '/' + info + '.txt'):
                        shutil.copyfile(self.datasetRoot + '/labels/' + self.curOpt + '/' + info + '.txt',
                                        self.outputRoot + '/labels/' + self.curOpt + '/' + info + '.txt')
                    continue

                # 处理原始图片信息

                img = cv2.imread(self.datasetRoot + '/images/' + self.curOpt + '/' + info + self.imgExe)  # 被处理的原始图片
                imgH = img.shape[0]
                imgW = img.shape[1]
                labels = []
                if os.path.exists(self.datasetRoot + '/labels/' + self.curOpt + '/' + info + '.txt'):
                    labelsStr = open(self.datasetRoot + '/labels/' + self.curOpt + '/' + info + '.txt', 'r').readlines()
                    for i in range(len(labelsStr)):
                        label = labelsStr[i]
                        items = label.strip().split(' ')
                        objClass = items[3]
                        xmin = int(items[4])
                        ymin = int(items[5])
                        xmax = int(items[6])
                        ymax = int(items[7])
                        labels.append([info, imgW, imgH, objClass, xmin, ymin, xmax, ymax])
                img, labels = self.pasteCropsOnImg(img, labels, newCropImgs, newCropImgsLabels, info)

                # 处理粘贴后结果
                fOut = open(self.outputRoot + '/labels/' + self.curOpt + '/' + info + '.txt', 'w')
                for label in labels:
                    if(self.showRect):
                        cv2.rectangle(img, (label[4], label[5]), (label[6], label[7]), (255, 0, 0))
                    fOut.write(
                        info + ' ' + str(imgW) + ' ' + str(imgH) + ' ' + str(label[3]) + ' ' + str(
                            label[4]) + ' ' + str(label[5]) + ' ' + str(label[6]) + ' ' + str(label[7]) + '\n')
                # cv2.imshow('aa', img)
                cv2.imwrite(self.outputRoot + '/images/' + self.curOpt + '/' + info + self.imgExe, img)

if __name__ == '__main__':
    datasetRoot = './dataTest_jing/'
    cropRoot = './dataTest_jing/objCrop/padding/1'
    outputRoot = './dataTest_jing/imgPaste/pasteRadio/pR3'
    imgNum = 1 # 数据集图片数量，不同数据集要改一下
    randomResizeRatioRange = [0.5, 1] # resizeRaion参数设置
    cropSizes = [[0, 416]] # 尺寸过滤器相关的参数，但跟论文里面反了一下，这边[0,416]是不过滤的尺寸
    cropClasses = [0,1,2] # 类型过滤器相关的参数，但跟论文里面反了一下，这边[0,1,2]是不过滤的类型
    imgExe = '.png' # 数据集图片的后缀
    pasteRadio = 3 # pasteRadio参数
    paste = Paste(datasetRoot, cropRoot, outputRoot, imgNum, imgExe=imgExe, pasteRadio=pasteRadio, cropSizes=cropSizes, cropClasses=cropClasses, showRect=False, randomResizeRatioRange=randomResizeRatioRange)
    paste.randomPaste()
    print(cropSizes)