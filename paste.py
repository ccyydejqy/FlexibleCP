import os
import cv2
import numpy as np
from until import until
import shutil
class Paste():
    def __init__(self, datasetRoot, cropRoot, outputRoot, imgNum, pasteNum=1, pasteRadio=1.0, imgExe='.png',
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

        self.pasteNum = pasteNum  # 如果这个值不为1，那么pasteRadio必须为1.0
        self.pasteRadio = pasteRadio  # 这个值只有pasteNum=1时才使用
        self.imgExe = imgExe
        self.imgSize = imgSize
        self.cropSizes = cropSizes
        self.cropClasses = cropClasses

        self.showRect = showRect
        self.randomResizeRatioRange = randomResizeRatioRange
        self.opts = opts
        self.curOpt = opts[0]


    def pasteCropsOnImg(self, img, labels, cropImgs, CropImgsLabels):
        imgW = img.shape[1]
        imgH = img.shape[0]
        hasPaste = False
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
            if(allowPatse == False): # 一张cropImg中有一个标签不符合尺寸过滤器，就不能粘贴
                continue
            hasPaste = True # 判断是否有cropImg最终能粘贴到原图中
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
            if(trying == True):
                return img, None
            img[pasteYmin:pasteYmax, pasteXmin:pasteXmax] = cropImg
            for cropImgLabel in cropImgLabels:
                appendArr = [labels[0][0], labels[0][1], labels[0][2], cropImgLabel[3], pasteXmin + cropImgLabel[4], pasteYmin + cropImgLabel[5], pasteXmin + cropImgLabel[6], pasteYmin + cropImgLabel[7]]
                # radio = min(self.imgSize[0] / imgW, self.imgSize[1] / imgH)
                # print((appendArr[6] - appendArr[4]) * (appendArr[7] - appendArr[5]) * radio * radio)
                labels.append(appendArr)
        if(hasPaste == False):
            return img, None
        return img, labels

    def createCropImgsAndImgsLabels(self, initCropIdx):
        cropImgsLen = len(self.files)
        pasteCropInfo = initCropIdx
        newCropImgsLabels = []
        newCropImgs = []
        resIdx = 0
        for pasteCount in range(self.pasteNum):
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
            pasteCropInfo = np.random.randint(0, cropImgsLen)  # 随机找另一种裁剪图
        return newCropImgs, newCropImgsLabels
    def randomPaste(self):
        for opt in self.opts:
            self.curOpt = opt
            os.makedirs(self.outputRoot + '/images/' + self.curOpt + '/')
            os.makedirs(self.outputRoot + '/labels/' + self.curOpt + '/')
            self.files = os.listdir(self.cropRoot + '/labels/' + self.curOpt + '/')
            cropImgsLen = len(self.files)  # 裁剪出来的图片的数目
            randomArr = np.random.randint(self.imgNum, size=cropImgsLen)
            for idx in range(int(cropImgsLen * self.pasteRadio)):
                # if(idx != 328):
                #     continue
                print(idx)
                cropInfo = self.files[idx].strip().split('.')[0]
                # 处理粘贴图片信息
                newCropImgs, newCropImgsLabels = self.createCropImgsAndImgsLabels(idx)
                if(len(newCropImgs) == 0):
                    continue
                tryPaste = True
                info = ''
                img = None
                labels = None
                imgW = imgH = 0
                tryNum = 0
                while (tryPaste and tryNum < 10):
                    tryNum += 1
                    info = str(randomArr[idx]).zfill(5)
                    # print(info)
                    # 处理被粘贴图片信息
                    img = cv2.imread(self.datasetRoot + '/images/' + self.curOpt + '/' + info + self.imgExe)  # 被处理的原始图片
                    imgH = img.shape[0]
                    imgW = img.shape[1]
                    if not os.path.exists(self.datasetRoot + '/labels/' + self.curOpt + '/' + info + '.txt'):
                        break
                    labelsStr = open(self.datasetRoot + '/labels/' + self.curOpt + '/' + info + '.txt', 'r').readlines()
                    labels = []
                    for i in range(len(labelsStr)):
                        label = labelsStr[i]
                        items = label.strip().split(' ')
                        objClass = items[3]
                        xmin = int(items[4])
                        ymin = int(items[5])
                        xmax = int(items[6])
                        ymax = int(items[7])
                        labels.append([info, imgW, imgH, objClass, xmin, ymin, xmax, ymax])
                    img, labels = self.pasteCropsOnImg(img, labels, newCropImgs, newCropImgsLabels)
                    if (labels != None):
                        tryPaste = False
                    else:
                        randomArr[idx] = np.random.randint(0, self.imgNum)
                if (tryPaste):
                    continue
                fOut = open(self.outputRoot + '/labels/' + self.curOpt + '/' + cropInfo + '_' + info + '.txt', 'w')
                for label in labels:
                    if(self.showRect):
                        cv2.rectangle(img, (label[4], label[5]), (label[6], label[7]), (255, 0, 0))
                    fOut.write(
                        cropInfo + '_' + info + ' ' + str(imgW) + ' ' + str(imgH) + ' ' + str(label[3]) + ' ' + str(
                            label[4]) + ' ' + str(label[5]) + ' ' + str(label[6]) + ' ' + str(label[7]) + '\n')
                # cv2.imshow('aa', img)
                cv2.imwrite(self.outputRoot + '/images/' + self.curOpt + '/' + cropInfo + '_' + info + self.imgExe, img)

if __name__ == '__main__':
    datasetRoot = '../CTSDB_jing/'
    cropRoot = './CTSDB_jing/objCrop/padding/0.5'
    outputRoot = './CTSDB_jing/imgPaste/cropSizes/16_64'
    imgNum = 700
    randomResizeRatioRange = [0.5, 1]
    cropSizes = [[16, 64]]
    cropClasses = [0, 1, 2]
    imgExe = '.png'
    paste = Paste(datasetRoot, cropRoot, outputRoot, imgNum, imgExe=imgExe,pasteNum=1, pasteRadio=1, cropSizes=cropSizes, cropClasses=cropClasses, showRect=False, randomResizeRatioRange=randomResizeRatioRange)
    paste.randomPaste()