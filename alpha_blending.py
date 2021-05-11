import cv2
import argparse

import numpy as np

if __name__ == "__main__":
    img_1 = cv2.imread("img_1.png") #画像の読み込み
    img_2 = cv2.imread("img_2.png")

    assert img_1.shape == img_2.shape #画像サイズが違う場合はエラーを返す。

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default = "alpha_blending", help = "average, emboss")

    args = parser.parse_args()

    if args.mode == 'alpha_blending': #アルファブレンディングの処理
        #縦方向のアルファブレンディング
        h = img_1.shape[0]
        alpha = np.linspace(0, 1, h).reshape(-1, 1, 1)  #画素ごとのαの値の生成
        height_alpha_img = img_1 * alpha + img_2 * (1 - alpha) 
        cv2.imwrite("height_alpha.png", height_alpha_img) #画像の保存

        #横方向のアルファブレンディング
        w = img_1.shape[1]
        alpha = np.linspace(0, 1, w).reshape(1, -1, 1) 
        width_alpha_img = img_1 * alpha + img_2 * (1 - alpha)
        cv2.imwrite("width_alpha.png", width_alpha_img)

    elif args.mode == "average": #画像の平均を得る処理
        ave_img = img_1 / 2 + img_2 / 2
        cv2.imwrite("average.png", ave_img)

    elif args.mode == 'emboss': #エンボスの処理
        #ネガポジ反転
        color_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2YCrCb)
        gray = color_img[:, :, 0]
        h, w = img_1.shape[:2]
        im_invert = cv2.bitwise_not(gray)

        #平行移動(アフィン変換)の処理
        tx, ty = 5, 5
        affine = np.float32([[1, 0, tx],[0, 1, ty]])
        img_afn = cv2.warpAffine(im_invert, affine, (w, h))

        #画像の合成
        emboss_img = gray + img_afn -128

        cv2.imwrite("emboss_img.png", emboss_img)