import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network_ResNet_ps_edge_1_new import Network
from utils.data_val import test_dataset
# 导入官方提供的评估函数
from utils.eval_functions import StructureMeasure, EnhancedMeasure, Fmeasure_calu

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='/root/reASGNet/Network_ResNet50/Net_epoch_best.pth')
parser.add_argument('--test_dataset_path', type=str, default='/root/reASGNet/data/TestDataset')
opt = parser.parse_args()

for _data_name in ['CVC-300', 'CVC-ColonDB']:
    data_path = opt.test_dataset_path + '/{}/'.format(_data_name)
    save_path = './results/{}/'.format(_data_name)
    os.makedirs(save_path, exist_ok=True)

    model = Network(channels=96)
    state_dict = torch.load(opt.pth_path)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()

    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    # 用于存储所有图片各项指标
    metrics_all = {'dice': [], 'iou': [], 'mae': [], 'sm': [], 'em': []}

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        result = model(image)

        res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        cv2.imwrite(save_path + name, res * 255)

        # ========== 开始官方标准的指标计算 ==========
        # 1. MAE
        mae = np.mean(np.abs(res - gt))

        # 2. Dice 和 IoU (官方代码里是通过调 Fmeasure_calu，通常取阈值为0.5，或者遍历阈值)
        # 为了公平，这里用0.5，你也可以改成 for t in np.arange(0.1, 1, 0.1) 取最大
        _, _, _, dice, _, iou = Fmeasure_calu(res, gt, threshold=0.5)

        # 3. S-measure 和 E-measure
        sm = StructureMeasure(res, gt)
        em = EnhancedMeasure(res, gt)

        metrics_all['dice'].append(dice)
        metrics_all['iou'].append(iou)
        metrics_all['mae'].append(mae)
        metrics_all['sm'].append(sm)
        metrics_all['em'].append(em)

        print('> {} - {}: Dice={:.4f}, IoU={:.4f}, MAE={:.4f}, S-measure={:.4f}, E-measure={:.4f}'.format(
            _data_name, name, dice, iou, mae, sm, em))

    # 打印整个数据集的平均指标
    print("\n" + "=" * 50)
    print(f"Dataset: {_data_name} - Summary")
    print(f"Average Dice:  {np.mean(metrics_all['dice']):.4f}")
    print(f"Average IoU:   {np.mean(metrics_all['iou']):.4f}")
    print(f"Average MAE:   {np.mean(metrics_all['mae']):.4f}")
    print(f"Average S-measure: {np.mean(metrics_all['sm']):.4f}")
    print(f"Average E-measure: {np.mean(metrics_all['em']):.4f}")
    print("=" * 50 + "\n")

# import torch
# import torch.nn.functional as F
# import numpy as np
# import os, argparse
# import cv2
# from lib.Network_ResNet_ps_edge_1_new import Network
# from utils.data_val import test_dataset
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--testsize', type=int, default=352, help='testing size')  #
# parser.add_argument('--pth_path', type=str,
#                     default='/root/reASGNet/Network_ResNet50/Net_epoch_best.pth')
# parser.add_argument('--test_dataset_path', type=str, default='/root/reASGNet/data/TestDataset')
# opt = parser.parse_args()
#
# for _data_name in ['CVC-300']:  # 'ETIS-LaribPolypDB','CVC-ClinicDB','Kvasir', 'CVC-ColonDB'
#     data_path = opt.test_dataset_path + '/{}/'.format(_data_name)
#     save_path = './results/'.format(opt.pth_path.split('/')[-2], _data_name)
#     os.makedirs(save_path, exist_ok=True)
#
#     model = Network(channels=96)
#     model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.pth_path).items()})
#     model.cuda()
#     model.eval()
#
#     image_root = '{}/images/'.format(data_path)
#     gt_root = '{}/masks/'.format(data_path)
#     test_loader = test_dataset(image_root, gt_root, opt.testsize)
#
#     for i in range(test_loader.size):
#         image, gt, name, _ = test_loader.load_data()
#         print('> {} - {}'.format(_data_name, name))
#
#         gt = np.asarray(gt, np.float32)
#         gt /= (gt.max() + 1e-8)
#         image = image.cuda()
#
#         result = model(image)
#
#         res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
#         res = res.sigmoid().data.cpu().numpy().squeeze()
#         res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#         cv2.imwrite(save_path + name, res * 255)
