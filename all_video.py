import numpy as np
import cv2
import sys
import math
    
def space(data2, j, x0, y0, x1, y1):
    # 各車両領域の外接矩形を青枠で表示
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0))
    # 各車両領域の重心座標を黒点で表示
    # cv2.circle(frame, (int(center2[j][0]), int(center2[j][1])), 1, (0, 0, 0), -1)
    # 車間距離を表示，各車両領域の外接矩形の枠の色を更新
    j_prev = j - 1
    if(0 <= j_prev):
        x0_prev = data2[j_prev][0] # 左上x座標
        y0_prev = data2[j_prev][1] # 左上y座標
        x1_prev = data2[j_prev][0] + data2[j_prev][2] # 右下x座標
        y1_prev = data2[j_prev][1] + data2[j_prev][3] # 右下y座標
        # 変換前後の対応点を設定
        p_original = np.float32([[447,578], [757,226], [428,142], [308,424], [370,145], [568,115]]) #背景画像
        p_trans = np.float32([[81,666], [327,397], [449,79], [89,576], [427,68], [504,90]]) #地図画像
        # 変換マトリクスと射影変換
        M, mask = cv2.findHomography(p_original, p_trans, 0)
        x_trans = (M[0,0] * ((x0 + x1) / 2) + M[0,1] * y0 + M[0,2]) / (M[2,0] * ((x0 + x1) / 2) + M[2,1] * y0 + M[2,2])
        y_trans = (M[1,0] * ((x0 + x1) / 2) + M[1,1] * y0 + M[1,2]) / (M[2,0] * ((x0 + x1) / 2) + M[2,1] * y0 + M[2,2])
        x_prev_trans = (M[0,0] * ((x0_prev + x1_prev) / 2) + M[0,1] * y1_prev + M[0,2]) / (M[2,0] * ((x0_prev + x1_prev) / 2) + M[2,1] * y1_prev + M[2,2])
        y_prev_trans = (M[1,0] * ((x0_prev + x1_prev) / 2) + M[1,1] * y1_prev + M[1,2]) / (M[2,0] * ((x0_prev + x1_prev) / 2) + M[2,1] * y1_prev + M[2,2])
        sqrt = math.sqrt((x_trans - x_prev_trans) * (x_trans - x_prev_trans) + (y_trans - y_prev_trans) * (y_trans - y_prev_trans))
        # cv2.line(gio, (x_prev_trans.astype(np.int64), y_prev_trans.astype(np.int64)), (x_trans.astype(np.int64), y_trans.astype(np.int64)), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        if(0 <= flag <= 3):
            if(sqrt < 63):
                if(flag == 0 or flag == 1):
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255))
                else:
                    cv2.rectangle(frame, (x0_prev, y0_prev), (x1_prev, y1_prev), (0, 0, 255))
            else:
                if(flag == 0 or flag == 1):
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0))
                else:
                    cv2.rectangle(frame, (x0_prev, y0_prev), (x1_prev, y1_prev), (0, 0, 0))
        elif(4 <= flag <= 8):
            if(sqrt < 56):
                if(flag == 4 or flag == 5):
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255))
                else:
                    cv2.rectangle(frame, (x0_prev, y0_prev), (x1_prev, y1_prev), (0, 0, 255))
            else:
                if(flag == 4 or flag == 5):
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0))
                else:
                    cv2.rectangle(frame, (x0_prev, y0_prev), (x1_prev, y1_prev), (0, 0, 0))

def labeling():
    #ラベリング
    dst_gs = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # ラベリング処理
    label = cv2.connectedComponentsWithStats(dst_gs)
    # オブジェクト情報を項目別に抽出
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    data2 = data
    # center = np.delete(label[3], 0, 0)
    # center2 = center
    mean = []
    j = 0
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    # オブジェクト情報を利用してラベリング結果を画面に表示
    for i in range(n):
        x0 = data[i][0] # 左上x座標
        y0 = data[i][1] # 左上y座標
        x1 = data[i][0] + data[i][2] # 右下x座標
        y1 = data[i][1] + data[i][3] # 右下y座標
        mean = np.append(mean, frame_lab[y0:y1, x0:x1, 0].mean()) 
        # 誤検出と影領域を除去
        if data[i][1] >= 380:
            if data[i][4] < 160 or mean[i] < 115: # lab 115 rgb 121
                # center2 = np.delete(center2, j, 0)
                data2 = np.delete(data2, j, 0)
                j -= 1
            else:
                space(data2, j, x0, y0, x1, y1)
        elif data[i][1] < 380 and data[i][1] >= 230:
            if data[i][4] < 95 or mean[i] < 115:
                # center2 = np.delete(center2, j, 0)
                data2 = np.delete(data2, j, 0)
                j -= 1
            else:
                space(data2, j, x0, y0, x1, y1)
        else:
            if data[i][4] < 40 or mean[i] < 115:
                # center2 = np.delete(center2, j, 0)
                data2 = np.delete(data2, j, 0)
                j -= 1
            else:
                space(data2, j, x0, y0, x1, y1)
        j += 1

def morphology():
    kernel = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=np.uint8)
    return cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

if __name__ == '__main__':
    cap = cv2.VideoCapture(f'../img_1fps/input.mp4')
    back = cv2.imread(f"../pre/back.png", cv2.IMREAD_COLOR)
    count = 0
    mask_color_list = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0], [128, 128, 128], [255, 255, 255]]
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(f'./all_video.mp4', fourcc, 30.0, (1280, 720))
    while True:
        ret, frame = cap.read()
        print(count)
        # マイナスの値を許容できる型に変換したうえで，引き算する
        subtracted = np.int16(frame) - np.int16(back)
        # そのうえで絶対値を取る
        subtracted = np.abs(subtracted).astype(np.uint8)
        subtracted_gs = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
        thresh, subtracted_bin = cv2.threshold(subtracted_gs, 0, 255, cv2.THRESH_OTSU)
        subtracted_bin_color = cv2.cvtColor(subtracted_bin, cv2.COLOR_GRAY2RGB)

        # マスキング(道路抽出)
        mask = cv2.imread(f"../tmp/tmp_4(video)/threshold.png", cv2.IMREAD_COLOR)
        for i in range(8):
            EXTRACT_MASK_COLOR = mask_color_list[i]
            mask_extracted = (np.all(mask == EXTRACT_MASK_COLOR, axis=2) * 255).astype(np.uint8)
            mask_extracted = (cv2.cvtColor(mask_extracted,  cv2.COLOR_GRAY2BGR) / 255).astype(np.uint8)
            dst = subtracted_bin_color * mask_extracted
            flag = i
            dst = morphology()
            labeling()
        if(count % 30 == 0):
            cv2.imwrite(f"./result_img/{count:03d}.png", frame)
        out.write(frame)
        count += 1

    out.release()
    cap.release()
    cv2.destroyAllWindows()