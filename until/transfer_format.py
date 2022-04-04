import os
import numpy as np
import cv2
import shutil
import xml.etree.ElementTree as ET
import xml.dom.minidom as xd

def yolov3_to_jing(yolov3Root, jingRoot, opts=None):
    '''
    这个函数是把标签的格式从yolov3源码格式转为jing自定义的格式，即[info, imgW, imgH, objClass, xmin, ymin, xmax, ymax]
    :param yolov3Root: 数据集目录，标签格式是yolov3源码的格式
    :param jingRoot: 格式转换后结果存放的跟目录
    :param opts: 需要转换训练集还是测试集，默认['test]
    :return:
    '''
    if opts is None:
        opts = ['test']
    if not os.path.exists(jingRoot + '/labels'):
        os.makedirs(jingRoot + '/labels')
    if not os.path.exists(jingRoot + '/images'):
        os.makedirs(jingRoot + '/images')
    for opt in opts:
        if(os.path.exists(jingRoot + '/labels/' + opt)):
            shutil.rmtree(jingRoot + '/labels/' + opt)
        os.makedirs(jingRoot + '/labels/' + opt)
        labelsRoot = yolov3Root + 'labels/' + opt + '/'
        imagesRoot = yolov3Root + 'images/' + opt + '/'
        files = os.listdir(imagesRoot)
        for file in files:
            info = file.strip().split('.')[0] # 00700
            if not (os.path.exists(labelsRoot + info + '.txt')):
                continue
            imgSuffix = file.strip().split('.')[1] # png
            img = cv2.imread(imagesRoot + file)
            imgH = img.shape[0]
            imgW = img.shape[1]

            fIn = open(labelsRoot + info + '.txt', 'r')
            labels = fIn.readlines()
            with open(jingRoot + '/labels/' + opt + '/' + info + '.txt', 'w') as fOut:
                for label in labels:
                    items = label.strip().split(' ')
                    objClass = items[0]
                    xc = int(float(items[1]) * imgW)
                    yc = int(float(items[2]) * imgH)
                    w = int(float(items[3]) * imgW)
                    h = int(float(items[4]) * imgH)
                    xmin = max(0, int(xc - w / 2))
                    ymin = max(0, int(yc - h / 2))
                    xmax = min(imgW, int(xc + w / 2))
                    ymax = min(imgH, int(yc + h / 2))
                    fOut.write(info + ' ' + str(imgW) + ' ' + str(imgH) + ' ' + objClass + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')

def jing_to_yolov3(jingRoot, yolov3Root, opts=None):
    if opts is None:
        opts = ['test']
    if not os.path.exists(yolov3Root + '/labels'):
        os.makedirs(yolov3Root + '/labels')
    if not os.path.exists(yolov3Root + '/images'):
        os.makedirs(yolov3Root + '/images')
    for opt in opts:
        if(os.path.exists(yolov3Root + '/labels/' + opt)):
            shutil.rmtree(yolov3Root + '/labels/' + opt)
        os.makedirs(yolov3Root + '/labels/' + opt)
        labelsRoot = jingRoot + '/labels/' + opt + '/'
        files = os.listdir(labelsRoot)
        for file in files:
            info = file.strip().split('.')[0] # 00700
            if not (os.path.exists(labelsRoot + info + '.txt')):
                continue

            fIn = open(labelsRoot + info + '.txt', 'r')
            labels = fIn.readlines()
            with open(yolov3Root + '/labels/' + opt + '/' + info + '.txt', 'w') as fOut:
                for label in labels:
                    items = label.strip().split(' ')
                    objClass = items[3]
                    imgW = int(items[1])
                    imgH = int(items[2])
                    xc = (float(items[6]) + float(items[4])) / 2 / imgW
                    yc = (float(items[7]) + float(items[5])) / 2 / imgH
                    w = (float(items[6]) - float(items[4])) / imgW
                    h = (float(items[7]) - float(items[5])) / imgH
                    xc = round(xc, 4)
                    yc = round(yc, 4)
                    w = round(w, 4)
                    h = round(h, 4)
                    fOut.write(objClass + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\n')



def VOC_to_jing(xmlDir, outputDir, opts=None):
    '''
    这个函数原始格式是VOC或者COCO的xml格式
    :param xmlDir: VOC数据集或者COCO2017数据集的xml文件所在的目录
    :param outputDir: 格式转换后结果存放的目录
    :param opts: 需要转换训练集还是测试集，默认['test]
    :return:
    '''
    if opts is None:
        opts = ['test']
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    for opt in opts:
        if (os.path.exists(outputDir + '/' + opt)):
            shutil.rmtree(outputDir + '/' + opt)
        os.makedirs(outputDir + '/' + opt)
        files = os.listdir(xmlDir + opt)
        for file in files:
            info = file.strip().split('.')[0]  # 00700
            if (os.path.exists(xmlDir + opt + '/' + file)):
                xml = xd.parse(xmlDir + opt + '/' + file)
                xml.root = xml.documentElement
                objects = xml.root.getElementsByTagName('object')
                names = xml.root.getElementsByTagName('name')
                width = xml.root.getElementsByTagName('width')[0].firstChild.data
                imgW = int(width)
                height = xml.root.getElementsByTagName('height')[0].firstChild.data
                imgH = int(height)
                objectsLen = objects.length
                if(objectsLen > 0):
                    with open(outputDir + opt + '/' +  info + '.txt', 'w') as fOut:
                        xmin_el = xml.root.getElementsByTagName('xmin')
                        ymin_el = xml.root.getElementsByTagName('ymin')
                        xmax_el = xml.root.getElementsByTagName('xmax')
                        ymax_el = xml.root.getElementsByTagName('ymax')
                        for i in range(objectsLen):
                            objClass = str(names[i].firstChild.data)
                            xmin = int(float(xmin_el[i].firstChild.data))
                            ymin = int(float(ymin_el[i].firstChild.data))
                            xmax = int(float(xmax_el[i].firstChild.data))
                            ymax = int(float(ymax_el[i].firstChild.data))
                            fOut.write(info + ' ' + str(imgW) + ' ' + str(imgH) + ' ' + objClass + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')

if __name__ == '__main__':
    # 原始的yolov3格式转我自己的格式
    # yolov3_to_jing('../CTSDB/', '../CTSDB_jing/', ['test'])

    # 增强结束后，要把我自己的格式转为Yolov3的格式
    dirName = 'pasteRadio'
    name = 'ORI'
    jing_to_yolov3('../GTSDB_jing/imgPaste/' + dirName + '/' + name,
                   '../GTSDB_yolov3/imgPaste_yolov3/' + dirName + '/' + name, ['test'])
    # if (os.path.exists('../GTSDB_yolov3/imgPaste_yolov3/' + dirName + '/' + name + '/images')):
    #     shutil.rmtree('../GTSDB_yolov3/imgPaste_yolov3/' + dirName + '/' + name + '/images')
    # shutil.move('../GTSDB_jing/imgPaste/' + dirName + '/' + name + '/images',
    #             '../GTSDB_yolov3/imgPaste_yolov3/' + dirName + '/' + name)