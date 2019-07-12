import os
import cv2
import numpy as np
from PIL import Image
from random import choice
from random import randint
from multiprocessing import Pool


def pil_image_rotate(pil_img):
    '''
    PIL图片旋转
    '''
    c = choice((1, 2, 3))
    angle = 0
    if c == 2:
        # 向左旋转
        angle = randint(0, 5)
    elif c == 3:
        # 向右旋转
        angle = randint(355, 360)
    return pil_img.rotate(angle, expand=True)


def pil_image_repair_square_and_white_bg(pil_img):
    '''
    PIL库Image填充成正方形和白色背景
    '''
    img = Image.new('RGB', (max(pil_img.size), max(pil_img.size)), (255, 255, 255))
    img.paste(pil_img, (0, 0, pil_img.size[0], pil_img.size[1]), pil_img)
    return img


def pil_image_valid_region(pil_img):
    '''
    获取PIL库Image图像有效区域
    此方法处理的图片应该是填充完背景为白色，并且补全为正方形的图片
    '''
    img = np.array(pil_img)
    xs = np.where(img[:,:,0]<255)[1]
    ys = np.where(img[:,:,0]<255)[0]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return (x, y, w, h)


def pil_image_translation(pil_img):
    '''
    PIL库Image对象平移
    '''
    # 图片的宽高
    img_w, img_h = pil_img.size
    # 图片内容的有效宽高
    x, y, w, h = pil_image_valid_region(pil_img)

    # 切割出有效部分
    valid_img = pil_img.crop((x, y, x+w, y+h))
    move_x = randint(0, img_w - w)
    move_y = randint(0, img_h - h)
    img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    img.paste(valid_img, (move_x, move_y))
    return img


def pil_image_resize(pil_img, size):
    '''
    PIL库Image对象重置大小
    '''
    img = pil_img.resize((size, size))
    return img


def pil_image_2_cv2_img(pil_img):
    '''
    PIL库Image对象转cv2的image(其实就是numpy的array)
    PIL的原本是四维的，而需求数据是三维的，所以转成了cv2的格式
    '''
    img = cv2.cvtColor(np.asarray(pil_img),cv2.COLOR_RGB2BGR)
    return img


def pil_image_amendment_2_cv2_img(pil_img):
    '''
    手写识别标注图片修正
    '''
    # 旋转
    pil_img = pil_image_rotate(pil_img)
    # 补全成正方形及填充白色背景
    pil_img = pil_image_repair_square_and_white_bg(pil_img)
    # 平移图像位置
    pil_img = pil_image_translation(pil_img)
    # 缩放到128
    pil_img = pil_image_resize(pil_img, 64)
    # 转成cv2
    img = pil_image_2_cv2_img(pil_img)
    return img


def proc_image(path, topath):
    pil_img = Image.open(path)
    img = pil_image_amendment_2_cv2_img(pil_img)
    cv2.imwrite(topath, img)


def apply(num):
    print(num)
    base_dir = './data/images/'
    img_dir = os.path.join(base_dir, str(num))
    images = os.listdir(img_dir)
    for image in images:
        img_path = os.path.join(img_dir, image)
        for i in range(100):
            topath = os.path.join(img_dir, image.split('.')[0]+'_'+str(i)+'.jpeg')
            proc_image(img_path, topath)
        os.remove(img_path)


if __name__ == '__main__':
    p = Pool(10)
    for i in range(10):
        p.apply_async(apply, args=(i,))
    p.close()
    p.join()
