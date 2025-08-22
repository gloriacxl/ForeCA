import torch
from torch import Tensor
from torchmetrics import Metric, PrecisionRecallCurve
final_out_mask_ = torch.rand(1, 1, 256, 256)  # 得到在 [0, 1)内的随机小数
final_out_mask_ = 0.5 * final_out_mask_ + 0.5  # 缩放到 [0, 1] 
#final_out_mask 就是最终得到的anomaly_map_score shape:[batchsize,1,256,256]
true_mask_show = torch.randint(low=0, high=2, size=(1, 1, 256, 256), dtype=torch.int) #true_mask_show就是groudtruth shape [1,1,256,256] 
#！！！！true_mask_show 一定要是int型，如果是float得改一下变量类型
final_label = torch.randint(low=0, high=2, size=(209, 1), dtype=torch.float) #image级别 假设209张照片 那会得到一个（209，1）的tensor 这是预测的
true_label = torch.randint(low=0, high=2, size=(209, 1), dtype=torch.int)    #groundtruth 也是（209，1）
def f1_score(preds,groudtruth):
    precision_recall_curve = PrecisionRecallCurve("binary")
    precision, recall, thresholds = precision_recall_curve(preds,groudtruth)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    threshold = thresholds[torch.argmax(f1_score)]
    optimal_f1_score = torch.max(f1_score)
    return optimal_f1_score.item()
    # print('optimal_f1_score=',optimal_f1_score)
#然后扔进f1_score函数里，第一个为预测值，第二个为groundtruth
#！！！！第一个类型为float，第二个为int！！！！
f1_score(final_out_mask_,true_mask_show)
f1_score(final_label,true_label)
