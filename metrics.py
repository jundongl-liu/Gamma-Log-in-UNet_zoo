import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
from PIL import Image
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def get_iou(mask_name,predict):
    
    #test_name = r"./saved_model/01014_mask.png"
    #image_test = cv2.imread(test_name, 0)


    image_mask = cv2.imread(mask_name,0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = cv2.resize(image_mask,(512,512))
    #print(image_mask.shape)
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height*weight)
    o = 0
    for row in range(height):
            for col in range(weight):
                if predict[row, col] < 0.5:  
                    predict[row, col] = 0
                else:
                    predict[row, col] = 1
                if predict[row, col] == 0 or predict[row, col] == 1:
                    o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    #print(height_mask,weight_mask)
    for row in range(height_mask):
            for col in range(weight_mask):
                if image_mask[row, col] < 1:   
                    image_mask[row, col] = 0
                else:
                    image_mask[row, col] = 1
                if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                    o += 1
    """
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_test[row, col] < 1:  
                image_test[row, col] = 0
            else:
                image_test[row, col] = 1
            if image_test[row, col] == 0 or image_test[row, col] == 1:
                o += 1
    """
    predict = predict.astype(np.int16)

    
    #interArea = np.multiply(image_test, image_mask)
    #tem = image_test + image_mask
    
    interArea = np.multiply(predict, image_mask)  #TP
    tem = predict + image_mask
    
    unionArea = tem - interArea  
    inter = np.sum(interArea)  #交集
    union = np.sum(unionArea)   #并集

    iou_tem = inter / union
    dif_area = union - inter
    dif_tem = (union - inter)/ union
    print('%s:iou=%f' % (mask_name,iou_tem))
    print('%s:dif_area=%f' % (mask_name, dif_area))
    print('%s:dif_tem=%f' % (mask_name, dif_tem))
    union_img = np.zeros((height,weight))
    inter_img = np.zeros((height,weight))
    for row in range(height):
        for col in range(weight):
            if image_mask[row, col] > 0 and image_test[row, col] > 0 :  #predict[row, col] > 0 :
                inter_img [row, col] = 1
            if image_mask[row, col] > 0 or image_test[row, col] > 0 :#predict[row, col] > 0 :
                union_img [row, col] = 1
    i_u_img = union_img - inter_img
    for row in range(height):
        for col in range(weight):
            if i_u_img[row, col] > 0 :
               i_u_img[row, col] = 0
            else :
                i_u_img[row, col] = 255

    return iou_tem,dif_tem,i_u_img

def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*precision*recall) / (precision+recall)    # F1
    return accuracy, precision, recall, F1


def get_dice(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = cv2.resize(image_mask,(512,512))
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                o += 1
    predict = predict.astype(np.int16)
    intersection = (predict*image_mask).sum()
    dice = (2. *intersection) /(predict.sum()+image_mask.sum())
    return dice

def get_hd(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    # print(mask_name)
    # print(image_mask)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = cv2.resize(image_mask,(512,512))
    #image_mask = mask
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                o += 1
    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = None
    if hd1>hd2 or hd1 == hd2:
        res=hd1
        return res
    else:
        res=hd2
        return res



def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()
