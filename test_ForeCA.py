import torch
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from model_unet_ori import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from random import choice
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc
import torch.nn.functional as F
from typing import Dict, List, Tuple
import pandas as pd
from statistics import mean as stat_mean
from numpy import ndarray
from skimage import measure
import re
from f1 import f1_score



size: int = 256
max_size: int = 512
mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


def test(obj_names, mvtec_path, checkpoint_path, save_name_, now_epoch):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    for obj_name in obj_names:
        device = torch.device("cuda:2")
        img_dim = 256
        save_name = save_name_

        model = ReconstructiveSubNetwork() # in_channels=3
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, save_name+".pckl"), map_location=device))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, save_name+"_seg.pckl"), map_location=device))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        #print(dataset)
        dataloader = DataLoader(dataset, batch_size = 1,
                                shuffle=False, num_workers = args.num_workers)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []


        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            with torch.no_grad():
                gray_rec, out_salient, _ = model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                out_salient_sm = torch.softmax(out_salient, dim=1)

            thre_list = [150,200,150,50,100,150,100,30,80,55,30,50]
            picked_thre = thre_list[int(args.obj_id)]
                
            for i in range(gray_batch.shape[0]):
                mask_array = np.array((out_salient_sm[i,1:,...]*255).detach().squeeze(0).cpu())
                mask_array[mask_array <= picked_thre] = 0 
                mask_array[mask_array > picked_thre] = 255

                final_out_mask = out_salient_sm*out_mask_sm           
                final_out_mask_ = final_out_mask[i,1:,...].detach().cpu().numpy()

            out_mask_bs = torch.tensor(final_out_mask_).unsqueeze(0) #刚好test的bs是1

            out_mask_cv = out_mask_bs[0 ,: ,: ,:].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_bs[: ,: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
                     
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)

        anomaly_score_prediction_tensor = torch.tensor(anomaly_score_prediction)
        anomaly_score_gt_tensor = torch.tensor(anomaly_score_gt.astype(int))
        image_f1 = f1_score(anomaly_score_prediction_tensor,anomaly_score_gt_tensor)
        
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("image_f1:  " +str(image_f1))
        print("==============================")

    print(obj_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))

    write_results_to_file(obj_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)
           
  
    competitive_indicator = torch.tensor(np.mean(obj_auroc_image_list))
    return competitive_indicator

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def __len__(self):
        return len(self.image_paths)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace
    import yaml

    parser = ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=5) # selected obj_id
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', action='store', type=float, default=0.0002)
    parser.add_argument('--epochs', action='store', type=int, default=500)
    parser.add_argument('--gpu_id', action='store', type=int, default=2)
    parser.add_argument('--log_path', action='store', type=str, default='./log/')
    parser.add_argument('--data_path', action='store', type=str, default=r'/home/dataset/VisA_pytorch/')   # your VisA path!
    parser.add_argument('--checkpoint_path', action='store', type=str, default='./checkpoint/') # your checkpoint path!
    parser.add_argument('--checkpoint_interval', type=int, default=50)  
    parser.add_argument('--image_visual_interval', type=int, default=100)  
    parser.add_argument("--config", type=str, default="./duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--suffix", type=str, default='')
    
    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.config}", 'r'))
    base_args.pop("dataset_name")
    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)
    torch.backends.cudnn.benchmark = False

    obj_batch = [['candle'],        #0
                 ['capsules'],      #1
                 ['cashew'],        #2
                 ['chewinggum'],    #3
                 ['fryum'],         #4
                 ['macaroni1'],     #5
                 ['macaroni2'],     #6
                 ['pcb1'],          #7
                 ['pcb2'],          #8
                 ['pcb3'],          #9
                 ['pcb4'],          #10
                 ['pipe_fryum']     #11
                 ]

    if int(args.obj_id) == -1:
        obj_list = ['candle',
                    'capsules',
                    'cashew',
                    'chewinggum',
                    'fryum',
                    'macaroni1',
                    'macaroni2',
                    'pcb1',
                    'pcb2',
                    'pcb3',
                    'pcb4',
                    'pipe_fryum'
                    ]

        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]


    run_name = 'ForeCA_' + str(picked_classes[0])
    picked_string = str(picked_classes[0])
    pattern = r'ForeCA_' + re.escape(picked_string) + r'_epoch(\d+)\.pckl\b'
    lunshu = [500]
    l = 0
    while l < len(lunshu):
        save_name = run_name + "_epoch" + str(lunshu[l])
        with torch.cuda.device(args.gpu_id):
            test(picked_classes, args.data_path, args.checkpoint_path, save_name, str(lunshu[l]))
        l = l + 1

