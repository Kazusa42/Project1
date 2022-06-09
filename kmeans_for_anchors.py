import glob
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def cas_iou(box, cls):
    x = np.minimum(cls[:, 0], box[0])
    y = np.minimum(cls[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cls[:, 0] * cls[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cls):
    return np.mean([np.max(cas_iou(box[idx], cls)) for idx in range(box.shape[0])])


def kmeans(box, k):
    row_num = box.shape[0]
    distance = np.empty((row_num, k))

    last_clu = np.zeros((row_num,))

    np.random.seed()
    cls = box[np.random.choice(row_num, k, replace=False)]

    iter_cnt = 0
    while True:
        for idx in range(row_num):
            distance[idx] = 1 - cas_iou(box[idx], cls)
        nearest = np.argmin(distance, axis=1)

        if (last_clu == nearest).all():
            break

        for j in range(k):
            cls[j] = np.median(
                box[nearest == j], axis=0)

        last_clu = nearest
        if iter_cnt % 5 == 0:
            print('iter: {:d}. avg_iou:{:.2f}'.format(iter_cnt, avg_iou(box, cls)))
        iter_cnt += 1

    return cls, nearest


def load_data(file_path):
    datas = []
    for xml_file in tqdm(glob.glob('{}/*xml'.format(file_path))):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height <= 0 or width <= 0:
            continue

        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            datas.append([xmax - xmin, ymax - ymin])
    return np.array(datas)


if __name__ == '__main__':
    np.random.seed(0)
    input_shape = [640, 640]
    anchors_num = 9
    path = './DOTA/Annotations'

    print('Load xmls.')
    data = load_data(path)
    print('Load xmls done.')

    print('K-means boxes.')
    cluster, near = kmeans(data, anchors_num)
    print('K-means boxes done.')
    data = data * np.array([input_shape[1], input_shape[0]])
    cluster = cluster * np.array([input_shape[1], input_shape[0]])

    for j in range(anchors_num):
        plt.scatter(data[near == j][:, 0], data[near == j][:, 1])
        plt.scatter(cluster[j][0], cluster[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print('Save kmeans_for_anchors.jpg in root dir.')

    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print('avg_ratio:{:.2f}'.format(avg_iou(data, cluster)))
    print(cluster)

    f = open("yolo_anchors.txt", 'w')
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()
