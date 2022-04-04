yolov3格式的数据集为dataTest，我自己格式的数据集为dataTest_jing。我所有的增强在我自己定义的格式上进行。

代码使用步骤：
1.  格式转换：   ./until/transfer_format.py 中调用yolov3_to_jing转换为jing格式
2.  目标裁剪：   执行crop.py   这个文件可以调padding
3.  目标粘贴：   执行paste2.py     这个文件可以调cropSizes（即尺寸过滤器）、cropClasses（类型过滤器）、
                                pasteRadio、randomResizeRatioRange（即resizeRadio）
4.  格式转换：   ./until/transfer_format.py 中调用jing_to_yolov3转换为yolov3格式

以上过程生成增强数据集，和原始数据集一起进行最终的训练