import os
from PIL import Image

root_dir = r"C:\Users\Kazusa\Desktop\DOTA_TRAIN_CROP\\"  # DOTA dataset path
annotations_dir = root_dir + r"labelTxt2/"
image_dir = root_dir + r"images/"
xml_dir = root_dir + r"XML/"  # output xml file path

"""
Script for converting DOTA annotation to VOC annotation
DOTA annotation format:
x1, y1, x2, y2, x4, y4, x3, y3, class_name, difficulty, (vertices are arranged in a clockwise order)
(x1, y1)-----------(x2, y2)
    |                  |
    |                  |
    |                  |
    |                  |
    |                  |
(x3, y3)-----------(x4, y4)
"""

for filename in os.listdir(annotations_dir):
    fin = open(annotations_dir + filename, 'r')
    image_name = filename.split('.')[0]
    img = Image.open(image_dir + image_name + ".jpg")
    xml_name = xml_dir + image_name + '.xml'
    with open(xml_name, 'w') as fout:
        fout.write('<annotation>' + '\n')

        fout.write('\t' + '<folder>DOTA 1.0</folder>' + '\n')
        fout.write('\t' + '<filename>' + image_name + '.jpg' + '</filename>' + '\n')

        fout.write('\t' + '<source>' + '\n')
        fout.write('\t\t' + '<database>' + 'DOTA 1.0' + '</database>' + '\n')
        fout.write('\t\t' + '<annotation>' + 'DOTA 1.0' + '</annotation>' + '\n')
        fout.write('\t\t' + '<image>' + 'flickr' + '</image>' + '\n')
        fout.write('\t\t' + '<flickrid>' + 'Unspecified' + '</flickrid>' + '\n')
        fout.write('\t' + '</source>' + '\n')

        fout.write('\t' + '<owner>' + '\n')
        fout.write('\t\t' + '<flickrid>' + 'Kazusa' + '</flickrid>' + '\n')
        fout.write('\t\t' + '<name>' + 'Kazusa' + '</name>' + '\n')
        fout.write('\t' + '</owner>' + '\n')

        fout.write('\t' + '<size>' + '\n')
        fout.write('\t\t' + '<width>' + str(img.size[0]) + '</width>' + '\n')
        fout.write('\t\t' + '<height>' + str(img.size[1]) + '</height>' + '\n')
        fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
        fout.write('\t' + '</size>' + '\n')

        fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

        for line in fin.readlines():
            line = line.split(' ')
            fout.write('\t' + '<object>' + '\n')
            fout.write('\t\t' + '<name>' + str(line[8]) + '</name>' + '\n')
            fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
            fout.write('\t\t' + '<truncated>' + line[6] + '</truncated>' + '\n')
            fout.write('\t\t' + '<difficult>' + str(int(line[9])) + '</difficult>' + '\n')
            fout.write('\t\t' + '<bndbox>' + '\n')
            fout.write('\t\t\t' + '<xmin>' + line[0] + '</xmin>' + '\n')
            fout.write('\t\t\t' + '<ymin>' + line[1] + '</ymin>' + '\n')
            # pay attention to this point!(0-based)
            fout.write('\t\t\t' + '<xmax>' + line[4] + '</xmax>' + '\n')
            fout.write('\t\t\t' + '<ymax>' + line[5] + '</ymax>' + '\n')
            fout.write('\t\t' + '</bndbox>' + '\n')
            fout.write('\t' + '</object>' + '\n')

        fin.close()
        fout.write('</annotation>')