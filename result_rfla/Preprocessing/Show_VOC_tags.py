# -*- coding: utf-8 -*-
#批量处理img和xml文件，根据xml文件中的坐标把img中的目标标记出来，并保存到指定文件夹，方便自己查看目标标记的是否准确。
import xml.etree.ElementTree as ET
import os, cv2
from tqdm import tqdm

annota_dir = r'/home/definfo/mmdet-rfla/result_rfla/Preprocessing/data_labeled_crop/Annotations'   #原始voc格式标签文件存放的文件夹，需改成自己的
origin_dir = r'/home/definfo/mmdet-rfla/result_rfla/Preprocessing/data_labeled_crop/JPEGImages'    #原始图片文件存放的文件夹，需改成自己的
target_dir1 = r'/home/definfo/mmdet-rfla/result_rfla/Preprocessing/plot_img'     #绘制完box后的图片存放的文件夹，需改成自己的

def divide_img(oriname):
    img_file = os.path.join(origin_dir, oriname + '.jpg')   #需要是jpg格式图片，如果你的是png或者其他格式，需改正
    im = cv2.imread(img_file)

    xml_file = os.path.join(annota_dir, oriname + '.xml')  # 读取每个原图像对应的xml文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
#im = cv2.imread(imgfile)
    for object in root.findall('object'):
        object_name = object.find('name').text
        Xmin = float(object.find('bndbox').find('xmin').text)
        Ymin = float(object.find('bndbox').find('ymin').text)
        Xmax = float(object.find('bndbox').find('xmax').text)
        Ymax = float(object.find('bndbox').find('ymax').text)
        color = (4, 250, 7)

        Xmin = int(max(Xmin, 0))
        Ymin = int(max(Ymin, 0))
        Xmax = int(min(Xmax, 639))
        Ymax = int(min(Ymax, 639))

        cv2.rectangle(im, (Xmin, Ymin), (Xmax, Ymax), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, object_name, (Xmin, Ymin - 7), font, 0.5, (6, 230, 230), 2)
        cv2.imshow('01', im)

    img_name = oriname + '.jpg'  #绘制box后的图片命名
    to_name = os.path.join(target_dir1, img_name)
    cv2.imwrite(to_name, im)

img_list = os.listdir(origin_dir)
for name in img_list:
    divide_img(name.rstrip('.jpg'))

