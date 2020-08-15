DATASET = "./HW2_Dataset"

base_img_name = "cyl_image0"
base_img_name2 = "cyl_image"
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import *

if __name__ == "__main__":
    for dir in os.listdir(DATASET):
        dir_path = DATASET + "/" + dir

        if os.path.isdir(dir_path):
            images = []
            acc_H = np.eye(3,3)
            for file in os.listdir(dir_path):
                images.append(file)
            images = sorted(images,key= lambda x: int(x[x.index(".")-2:x.index(".")]))
            img1_path = dir_path + "/" + images[0]
            img1 = cv2.imread(img1_path,0)
            img1 = cv2.resize(img1, (480, 479), interpolation=cv2.INTER_AREA)
            gt_image = cv2.imread(DATASET + "/" +dir + "_gt.png")
            gt_image = gt_image[:,:3*480]

            for i in range(1,8):
                img2_path = dir_path + "/" + images[i]
                img2 = cv2.imread(img2_path,0)
                img2 = cv2.resize(img2, (480,479), interpolation = cv2.INTER_AREA)
                if i != 1:
                    kp1,kp2,good = get_matches(img1[:,-img2.shape[1]-100:],img2)
                else:
                    kp1,kp2,good = get_matches(img1,img2)

                pt1,pt2 = getPoints(kp1,kp2,good)
                H = findHomography(good,pt1,pt2)
                height, width = img2.shape
                new_H = getHomCorrection(width,height,H)
                acc_H = new_H
                myReg = warpImage(new_H,img1,img2)

                best_width = findAlignment(img1,myReg)
                #print(new_H)
                mergedImages = mergeImages(img1,myReg,best_width)
                img1 = mergedImages
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax1.imshow(img1,cmap="gray")
            ax1.title.set_text("merged")
            ax2 = fig.add_subplot(2,1,2)
            ax2.title.set_text("Ground Truth")
            ax2.imshow(gt_image,cmap="gray")
            plt.pause(0.01)



