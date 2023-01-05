import cv2
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

def match(img1,img2):
    (h,w) = img1.shape[:2]
    resized = cv2.resize(img2,(w,h))
    (h1,w1) = resized.shape[:2]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return ssim(img1, img2)

#读取图片
def read_directory(directory_name, array_of_img):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
    return array_of_img

if __name__ == '__main__':
    # imfil1 = 'ori51/rb2d_ra1e6_s42/frames_b_gt/frame_157.png'
    # imfil2 = 'ori51/rb2d_ra1e6_s42/frames_b_pred/frame_157.png'
    # # imfil1 = 'octree_v3/rb2d_ra1e6_s42/frames_b_gt/frame_157.png'
    # # imfil2 = 'octree_v3/rb2d_ra1e6_s42/frames_b_pred/frame_157.png'
    # # imfil1 = 'new51/rb2d_ra1e6_s42/frames_b_gt/frame_157.png'
    # # imfil2 = 'new51/rb2d_ra1e6_s42/frames_b_pred/frame_157.png'
    # loss = match(imfil1,imfil2)
    # print(loss)
    img_dic = {}
    dir = 'v8/rb2d_ra1e6_s42/SSIM/'
    for dir_name in os.listdir(r"./" + dir):
        array_of_img = []
        dir_name_new = dir + dir_name
        temp_img = read_directory(dir_name_new,array_of_img)
        img_dic[dir_name] = temp_img

    gt_name = []
    pred_name = []
    count = 1
    for key in img_dic.keys():
        if count%2 != 0:
            gt_name.append(key)
        if count%2 == 0:
            pred_name.append(key)
        count += 1

    sum_loss = 0
    for i in range(len(gt_name)):
        for j in range(len(img_dic[gt_name[i]])):
            img_gt = img_dic[gt_name[i]][j]
            img_pred = img_dic[pred_name[i]][j]
            loss = 1. - match(img_gt, img_pred)
            sum_loss += loss

    print(sum_loss)