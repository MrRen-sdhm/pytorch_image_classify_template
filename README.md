# **pytorch_image_classify_template**

use pytorch to do image classfiication tasks 

reference: https://github.com/spytensor/pytorch-image-classification

```bash
loading train dataset
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4572/4572 [00:00<00:00, 1145746.42it/s]
loading train dataset
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2520/2520 [00:00<00:00, 1128994.45it/s]
Train Epoch:  1/40 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]  [Current: Loss 1.079473 Top1: 75.131233 ]  286/286 [ 100% ]
Val   Epoch:  1/40 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]  [Current: Loss 0.469368 Top1: 89.484131 ]  79/79 [ 100% ]
Get Better top1 : 89.4841 saving weights to ./checkpoints/best_model/resnet50/0/model_best.pth.tar
Train Epoch:  2/40 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]  [Current: Loss 0.696026 Top1: 83.442696 ]  286/286 [ 100% ]
Val   Epoch:  2/40 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]  [Current: Loss 0.275893 Top1: 91.269844 ]  79/79 [ 100% ]
Get Better top1 : 91.2698 saving weights to ./checkpoints/best_model/resnet50/0/model_best.pth.tar
Train Epoch:  3/40 [>>>>>                                             ]  [Current: Loss 0.667610 Top1: 84.165291 ]  31/286 [  10% ]
```
