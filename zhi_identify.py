# encoding="utf-8"
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from models.pfld import PFLDInference
from mtcnn.detector import detect_faces


def isRayIntersectsSegment(poi, s_poi, e_poi):  # [x,y] [lng,lat]
    # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
        return False

    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


def isPoiWithinPoly(poi, poly):
    # 输入：点，多边形三维数组
    # poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    # 可以先判断点是否在外包矩形内
    # if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    # 但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc = 0  # 交点个数
    for epoly in poly:  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly) - 1):  # [0,len-1]
            s_poi = epoly[i]
            e_poi = epoly[i + 1]
            if isRayIntersectsSegment(poi, s_poi, e_poi):
                sinsc += 1  # 有交点就加1
    return True if sinsc % 2 == 1 else False


# 判断是否落在眉毛上
def point_in_eyebrow(keypoints, zhi):
    left_eyebrows = [keypoints[33:42]]
    right_eyebrows = [keypoints[42:51]]
    if isPoiWithinPoly(zhi, left_eyebrows):
        return True
    elif isPoiWithinPoly(zhi, right_eyebrows):
        return True
    else:
        return False


# 判断黑色点是否在眼睛内
def point_in_eyes(keypoints, zhi):
    left_eye = [keypoints[60:68]]
    right_eye = [keypoints[68:76]]
    if isPoiWithinPoly(zhi, left_eye):
        return True
    elif isPoiWithinPoly(zhi, right_eye):
        return True
    else:
        return False


# 判断黑色点是否在鼻孔内
def point_in_nostril(keypoints, zhi):
    nostril = [keypoints[53:60]]
    if isPoiWithinPoly(zhi, nostril):
        return True
    else:
        return False


# 判断是否在嘴巴上
def point_in_mouth(keypoints, zhi):
    mouth = [keypoints[76:88]]
    if isPoiWithinPoly(zhi, mouth):
        return True
    else:
        return False


# 判断是否在面部轮廓内部
def point_in_face(keypoints, zhi):
    face = keypoints[0:38]
    face.extend(keypoints[42:47])
    face = [face]
    if isPoiWithinPoly(zhi, face):
        return True
    else:
        return False


# 提取面部98个关键点
def extract_keypoints(img_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    img = cv2.imread(img_path, 1)
    height, width = img.shape[:2]
    bounding_boxes, landmarks = detect_faces(img)
    for box in bounding_boxes:
        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1  # 宽度
        h = y2 - y1 + 1  # 高度
        cx = x1 + w // 2  # 中心宽度
        cy = y1 + h // 2  # 中心高度

        size = int(max([w, h]) * 1.1)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        edx1 = max(0, -x1)
        edy1 = max(0, -y1)
        edx2 = max(0, x2 - width)
        edy2 = max(0, y2 - height)

        cropped = img[y1:y2, x1:x2]
        if edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0:
            cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                         cv2.BORDER_CONSTANT, 0)
        input = cv2.resize(cropped, (112, 112))
        cv2.imwrite(img_path.replace(".png", "_1.png"), input)
        input = transform(input).unsqueeze(0).to(device)
        _, landmarks = pfld_backbone(input)
        pre_landmark = landmarks[0]
        key_points = pre_landmark.cpu().detach().numpy().reshape(
            -1, 2) * [size, size] - [edx1, edy1]
        keypoints = []
        for (x, y) in key_points:
            cv2.circle(img, (x1 + int(x), y1 + int(y)), 2, (0, 255, 0), -1)
            keypoints.append([x1 + int(x), y1 + int(y)])
    cv2.imshow('face_landmark_68', img)
    cv2.waitKey(1000)
    return keypoints


def extract_candidate_zhis(img_path, radius):
    # img_path=img_path.replace(".png","_1.png")
    img = Image.open(img_path)
    img1 = cv2.imread(img_path, 1)
    pixels = img.load()
    zhis = []
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            # 每个像素点的r、g、b分量中的最大值作为该像素点的亮度值
            pixels[x, y] = max(pixels[x, y])
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            left_y = y - radius if y - radius > 0 else 0
            right_y = y + radius if y + radius < img.size[1] else img.size[1]
            left_x = x - radius if x - radius > 0 else 0
            right_x = x + radius if x + radius < img.size[0] else img.size[0]
            pixel_x_y = []
            # 收集该点相邻3个点的像素值
            for i in range(left_y, right_y):
                for j in range(left_x, right_x):
                    pixel_x_y.append(pixels[j, i][0])
            # print(pixel_x_y,pixels[x,y])
            pixel_mean = np.mean(pixel_x_y)
            delta = pixels[x, y][0] - pixel_mean
            # print(delta)
            # (46,73) (25,36) (36,68) (61,89)
            # if x == 61 and y == 89:
            #     print("观察值{}".format(delta))
            if delta < -80:
                zhis.append([x, y])
                cv2.circle(img1, (x, y), 1, (0, 255, 0))
    cv2.imshow("test", img1)
    cv2.waitKey(1000)
    return zhis


if __name__ == "__main__":
    img_path = "../../data/mini_test/88_1.png"
    model_path = "../../checkpoint/snapshot/checkpoint.pth.tar"
    keypoints = extract_keypoints(img_path, model_path)
    zhis = extract_candidate_zhis(img_path, 3)
    candidate = zhis.copy()
    for i in zhis:
        if point_in_eyes(keypoints, i):
            if i in candidate:
                candidate.remove(i)
        if point_in_eyebrow(keypoints, i):
            if i in candidate:
                candidate.remove(i)
        if point_in_nostril(keypoints, i):
            if i in candidate:
                candidate.remove(i)
        if point_in_mouth(keypoints, i):
            if i in candidate:
                candidate.remove(i)
        # if not point_in_face(keypoints, i):
        #     if i in candidate:
        #         candidate.remove(i)
    img = cv2.imread(img_path, 1)
    for x, y in candidate:
        cv2.circle(img, (x, y), 1, (0, 255, 0))
    cv2.imshow("test", img)
    cv2.waitKey(100000)
