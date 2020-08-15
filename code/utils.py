import numpy as np
import cv2
from matplotlib import pyplot as plt
def get_matches(img1,img2):

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:

        if m.distance < 0.65 * n.distance:
            good.append(m)
    #good = bf.match(des1, des2)

    keypoints = cv2.drawKeypoints(img1,kp1,None,flags=2)
    keypoints2 = cv2.drawKeypoints(img2,kp2,None,flags=2)

    match_points = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(keypoints)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(keypoints2)
    ax1.title.set_text("Left  image with keypoints")
    ax2.title.set_text("Right image with keypoints")
    plt.pause(0.01)
    plt.imshow(match_points)
    plt.pause(0.01)

    return kp1,kp2,good

def getPoints(kp1,kp2,good):
    pt1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pt2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pt1,pt2

def getSVDmatrix(matches):
    if len(matches) >= 0:
        result = []
        for point1, point2 in matches:
            x, y = point1
            xx, yy = point2
            ithpt = [[x, y, 1.0, 0.0, 0.0, 0.0, -xx * x, -xx * y, -xx],
             [0.0, 0.0, 0.0, -x, -y, -1.0, yy * x, yy * y, yy]]
            if result == []:
                result = ithpt
            else:
                result = np.vstack((result,ithpt))
        return result
    else:
        return None


def findHomography(matches, kp1, kp2, num=8, num_iters=10000, threshold=1):
    best = 0
    best_H = []
    for i in range(num_iters):
        indices = np.random.choice(len(matches), num, replace=True)
        temp_matches = []
        for ind in indices:
            temp_matches.append([kp1[ind], kp2[ind]])
        P = getSVDmatrix(temp_matches)
        A, B, C = np.linalg.svd(P)
        H = C[-1, :].reshape((3, 3))

        H = (1 / H.item(8)) * H

        H_query = np.dot(np.hstack((kp1, np.ones((kp1.shape[0], 1)))), H)
        H_query = H_query[:, :-1]
        p1 = np.hstack((kp2, np.ones((kp2.shape[0], 1))))
        p2 = np.hstack((kp1, np.ones((kp1.shape[0], 1))))
        p1hat = np.dot(H, p2.T).T
        W = p1hat[:, -1].reshape(p1hat.shape[0], 1).repeat(3, axis=-1)
        p1hat = p1hat / W
        diffs = np.linalg.norm(p1hat - p1, axis=1)
        count = len(np.where(diffs < threshold)[0])
        if count > best:
            best = count
            best_H = H
            #print(best)
            #print("woa")

    return best_H


def getHomCorrection(width, height, H):
    P = [[0, width, width, 0],
         [0, 0, height, height],
         [1, 1, 1, 1]]
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pus = np.dot(H, P)
    pus[0] = pus[0] / pus[2]
    pus[1] = pus[1] / pus[2]
    minx, miny = min(pus[0]), min(pus[1])
    if minx < 0:
        A[0,2] = -minx
    if miny < 0:
        A[1,2] = -miny
    best_A = np.dot(A, H)
    return best_A


def findAlignment(img1, img2):
    height, width = img2.shape
    best_width = 0
    best_sqd = 9999999
    widths = []
    sqds = []
    regions = []
    while width >= img1.shape[1]:
        test_region = img2[:, width - img1.shape[1]:width]

        sqd = np.sqrt(np.sum(np.square(test_region - img1)))
        sqds.append(sqd)
        widths.append(width)
        regions.append(test_region)
        if sqd < best_sqd:
            best_sqd = sqd
            best_width = width

        width = width - 1

    return best_width
def warpImage(H,img1,img2):
    width, height = img2.shape
    result = np.zeros((width, height))
    indices = np.indices((width, height))
    rows = indices[0].reshape(indices[0].shape[0] * indices[0].shape[1])
    cols = indices[1].reshape(indices[1].shape[0] * indices[1].shape[1])
    ones = np.ones(rows.shape[0])
    inds = np.vstack((rows, cols, ones))
    warped_inds = np.dot(np.linalg.inv(H), inds)
    warped_inds[0], warped_inds[1] = warped_inds[0] / warped_inds[2], warped_inds[1] / warped_inds[2]
    warped_inds = np.int32(np.rint(warped_inds[:2]))
    warped_inds = (warped_inds[0], warped_inds[1])
    nonzero_inds1 = np.where(np.logical_and(np.logical_and(np.logical_and((warped_inds[0] >= 0) ,(warped_inds[1] >= 0)) ,(warped_inds[0] <result.shape[0])), (warped_inds[1] < result.shape[1])))
    non_zero_rows = warped_inds[0][nonzero_inds1]
    non_zero_cols = warped_inds[1][nonzero_inds1]
    warped_inds = (non_zero_rows,non_zero_cols)

    img_inds = (np.int32(inds[0][nonzero_inds1]), np.int32(inds[1][nonzero_inds1]))
    result[img_inds] = img2[warped_inds]
    zeros = np.zeros((img1.shape[0],img1.shape[1]+20))
    result = np.hstack((zeros,result))
    return result.astype("uint8")
def mergeImages(img1,img2,best_width):
    w = best_width
    x = img2.copy()

    p = x[:, w - img1.shape[1]:w].copy()

    p[:img1.shape[0], :img1.shape[1]] = img1

    res = np.hstack((p,x[:, w::]))
    return res
