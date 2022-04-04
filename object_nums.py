# 这个文件分析不同尺寸目标数目
import os
import numpy as np
import cv2
class ObjuctNums():
    def __init__(self, datasetRoot, objSizes, imgSize=None, opts=None):
        if opts is None:
            opts = ['test']
        if imgSize is None:
            imgSize = [416, 416]
        self.datasetRoot = datasetRoot
        self.objSizes = objSizes
        self.imgSize = imgSize
        self.opts = opts
        self.res = {}
    def solveCTSDB_jing(self, classNum=3):
        for opt in opts:
            files = os.listdir(self.datasetRoot + '/labels/' + opt)
            self.res[opt] = np.zeros([classNum, len(objSizes)])
            for file in files:
                fIn = open(self.datasetRoot + '/labels/' + opt + '/' + file, 'r')
                labels = fIn.readlines()
                for label in labels:
                    items = label.strip().split(' ')
                    imgW = int(items[1])
                    imgH = int(items[2])
                    className = int(items[3])
                    radio = min(imgSize[0] / imgW, imgSize[1] / imgH)
                    ymax = int(items[-1])
                    xmax = int(items[-2])
                    ymin = int(items[-3])
                    xmin = int(items[-4])
                    labelW = (xmax - xmin) * radio
                    labelH = (ymax - ymin) * radio
                    labelWH = int(labelW * labelH)
                    for i in range(len(objSizes)):
                        objSize = objSizes[i]
                        if(objSize[0] * objSize[0] < labelWH and labelWH <= objSize[1] * objSize[1]):
                            self.res[opt][className][i] += 1
                            break
        return self.res
    def solveCTSDB_yolov3(self, classNum=3):
        for opt in opts:
            files = os.listdir(self.datasetRoot + '/labels/' + opt)
            self.res[opt] = np.zeros([classNum, len(objSizes)])
            for file in files:
                info = file.split('.')[0]
                img = cv2.imread(self.datasetRoot + '/images/' + opt + '/' + info + '.png')
                imgW = img.shape[1]
                imgH = img.shape[1]
                fIn = open(self.datasetRoot + '/labels/' + opt + '/' + file, 'r')
                labels = fIn.readlines()
                for label in labels:
                    items = label.strip().split(' ')
                    className = int(items[0])
                    xc = int(float(items[1]) * imgW)
                    yc = int(float(items[2]) * imgH)
                    w = int(float(items[3]) * imgW)
                    h = int(float(items[4]) * imgH)
                    radio = min(self.imgSize[0] / imgW, self.imgSize[1] / imgH)
                    xmin = max(0, xc - w // 2)
                    ymin = max(0, yc - h // 2)
                    xmax = min(imgW, xc + w // 2)
                    ymax = min(imgH, yc + h // 2)
                    labelW = (xmax - xmin) * radio
                    labelH = (ymax - ymin) * radio
                    labelWH = int(labelW * labelH)
                    for i in range(len(objSizes)):
                        objSize = objSizes[i]
                        if(objSize[0] * objSize[0] < labelWH and labelWH <= objSize[1] * objSize[1]):
                            self.res[opt][className][i] += 1
                            break
        return self.res
    def solve(self):
        for opt in opts:
            files = os.listdir(self.datasetRoot + '/' + opt)
            self.res[opt] = np.zeros(len(objSizes))
            for file in files:
                fIn = open(self.datasetRoot + '/' + opt + '/' + file, 'r')
                labels = fIn.readlines()
                for label in labels:
                    items = label.strip().split(' ')
                    imgW = int(items[1])
                    imgH = int(items[2])
                    radio = min(imgSize[0] / imgW, imgSize[1] / imgH)
                    ymax = int(items[-1])
                    xmax = int(items[-2])
                    ymin = int(items[-3])
                    xmin = int(items[-4])
                    labelW = (xmax - xmin) * radio
                    labelH = (ymax - ymin) * radio
                    labelWH = int(labelW * labelH)
                    for i in range(len(objSizes)):
                        objSize = objSizes[i]
                        if(objSize[0] * objSize[0] < labelWH and labelWH <= objSize[1] * objSize[1]):
                            self.res[opt][i] += 1
                            break
        return self.res

if __name__ == '__main__':
    dataset = '../CTSDB_jing/'
    # dataset = './CTSDB_jing/imgPaste/trying'

    imgSize = [416, 416]
    opts = ['train','test']
    objSizes = []
    for i in range(200):
        objSizes.append([i, i + 1])
    # objSizes = [[0,8], [0, 16], [16, 32],[32,64],[64,128], [128,500]]
    aaa = ObjuctNums(dataset, objSizes, imgSize, opts)
    res = aaa.solveCTSDB_jing(4)
    resTrain = res['train']
    resTest = res['test']
    resAll = resTrain + resTest
    # print(resAll[0])
    pre = [0,0,0,0]
    diejia = resAll
    for i in range(200):
        for j in range(4):
            diejia[j][i] = diejia[j][i] + pre[j]
            pre[j] = diejia[j][i]
    final = diejia[0] + diejia[1] + diejia[2] + diejia[3]
    final = final / final[-1]
    idx = 0
    for i in range(len(final)):
        idx = i
        if(final[i] == 1):
            break
    final = final[0:idx+1]
    print(len(final))
    print(final)