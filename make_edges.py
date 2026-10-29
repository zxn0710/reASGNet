import os
import cv2
import numpy as np


def generate_edges(mask_dir, edge_dir):
    os.makedirs(edge_dir, exist_ok=True)
    for name in os.listdir(mask_dir):
        if not (name.endswith('.png') or name.endswith('.jpg')):
            continue
        mask_path = os.path.join(mask_dir, name)
        mask = cv2.imread(mask_path, 0)

        # 核心边缘生成逻辑（照搬作者的）
        body = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
        body = body ** 0.5
        tmp = body[np.where(body > 0)]
        if len(tmp) != 0:
            a = np.where(body > 2)
            body[a] = 255
        res = mask - body
        res = cv2.blur(res, (3, 3))

        # 保存
        save_path = os.path.join(edge_dir, name)
        cv2.imwrite(save_path, res)


if __name__ == '__main__':
    # 修改成你自己的真实路径！！
    TRAIN_MASK_DIR = '/root/reASGNet/data/TrainDataset/masks/'
    TRAIN_EDGE_DIR = '/root/reASGNet/data/TrainDataset/edges/'

    print(f"正在从 {TRAIN_MASK_DIR} 生成边缘图到 {TRAIN_EDGE_DIR} ...")
    generate_edges(TRAIN_MASK_DIR, TRAIN_EDGE_DIR)
    print("生成完毕！")