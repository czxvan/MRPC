import cv2
import numpy as np
import os

GREEN = [112, 236, 160]
GREY = [235, 237, 238]
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

WINDOW_BGCOLOR = GREY
SELF_MSG_BGCOLOR = np.array(GREEN)
OTHER_MSG_BGCOLOR = WHITE
TEXT_COLOR = BLACK

def cleanDir(dir_path):
    for filename in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, filename))

def change_color(image, color1, color2, threshold=10):
    # 单通道
    if isinstance(color1, int):
        mask = cv2.inRange(image, color1 - threshold, color1 + threshold)
        image[mask > 0] = color2
    # 三通道
    else:
        threshold = np.array([threshold, threshold, threshold])
        mask = cv2.inRange(image, np.array(color1) - threshold, np.array(color1) + threshold)
        image[mask > 0] = np.array(color2)

def extract_squares(image):
    change_color(image, WINDOW_BGCOLOR, [0, 0, 0], threshold=15)
    squares = []
    contours,_ = cv2.findContours(image[:, :, 2],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        cv2.drawContours(image, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        if area > 5000:
            x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度
            w,h = 115, 115 #设置固定宽高[TODO]
            squares.append((x, y, w, h))
    
    # 按照x坐标排序
    squares.sort(key=lambda y: y[1])
    return squares

def compare_similarity(imageA, imageB):
    # 计算直方图
    histA = cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # 归一化直方图
    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)
    # 计算相似度
    similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

    return similarity

def deduplicate_images(images, threshold=0.9):
    unique_images = []
    for image in images:
        is_duplicate = False
        for unique_image in unique_images:
            # 计算相似度
            similarity = compare_similarity(image, unique_image)
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_images.append(image)
    return unique_images

    
def get_avtar_items(img):
    # 清空文件夹
    cleanDir('images/left_avatars')
    cleanDir('images/right_avatars')

    # 获取头像
    left_part = img[:, :170, :]
    right_part = img[:, 1030:1180, :]

    left_squares = extract_squares(left_part.copy())
    right_squares = extract_squares(right_part.copy())

    left_avatars = []
    for square in left_squares:
        left_avatars.append(left_part[square[1]:square[1] + square[3], square[0]:square[0] + square[2]])
    right_avatars = []
    for square in right_squares:
        right_avatars.append(right_part[square[1]:square[1] + square[3], square[0]:square[0] + square[2]])
    
    unique_left_avatars = deduplicate_images(left_avatars, threshold=0.7)
    unique_right_avatars = deduplicate_images(right_avatars, threshold=0.7)
    
    for i,avatar in enumerate(unique_left_avatars):
        cv2.imwrite(f'images/left_avatars/{i}.jpg', avatar)
    for i,avatar in enumerate(unique_right_avatars):
        cv2.imwrite(f'images/right_avatars/{i}.jpg', avatar)


# 读取图片
src_image = cv2.imread('testcases/2.jpg')
# 删去上下多余的部分
h, w, _ = src_image.shape
img = src_image.copy()[230:h-120, :, :]
get_avtar_items(img)
