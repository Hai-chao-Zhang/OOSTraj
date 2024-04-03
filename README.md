<div align="center">

# OOSTraj

## Out-of-Sight Trajectory Prediction 

### CVPR24

by [Haichao Zhang](https://Hai-chao-Zhang.github.io/), [Yi Xu](https://sites.google.com/view/homepage-of-yi-xu/), 
[Hongsheng Lu](https://www.linkedin.com/in/hongsheng-lu-178486102/), [Takayuki Shimizu](https://www.linkedin.com/in/takashimizu/), [Yun Fu](http://www1.ece.neu.edu/~yunfu/). 
</div>

---

Pytorch implementation of our method "OOSTraj: Out-of-Sight Trajectory Prediction With Vision-Positioning Denoising" in CVPR2024.

<div align="center">

<a href="">
<img width="800" alt="image" src="https://www.zhanghaichao.xyz/Out-of-SightTrajPred/assets/head.png">
</a>
</div>

---

Trajectory prediction is fundamental in computer vision and autonomous driving, particularly for understanding pedestrian behavior and enabling proactive decision-making. Existing approaches in this field often assume precise and complete observational data, neglecting the challenges associated with out-of-view objects and the noise inherent in sensor data due to limited camera range, physical obstructions, and the absence of ground truth for denoised sensor data. Such oversights are critical safety concerns, as they can result in missing essential, non-visible objects. To bridge this gap, we present a novel method for out-of-sight trajectory prediction that leverages a vision-positioning technique. Our approach denoises noisy sensor observations in an unsupervised manner and precisely maps sensor-based trajectories of out-of-sight objects into visual trajectories. This method has demonstrated state-of-the-art performance in out-of-sight noisy sensor trajectory denoising and prediction on the Vi-Fi and JRDB datasets. By enhancing trajectory prediction accuracy and addressing the challenges of out-of-sight objects, our work significantly contributes to improving the safety and reliability of autonomous driving in complex environments. Our work represents the first initiative towards Out-Of-Sight Trajectory prediction (OOSTraj), setting a new benchmark for future research. 

---

<div align="center">
"Vision-Postioning Denoising Model" inside OOSTraj

<a href="">
<img width="800" alt="image" src="https://www.zhanghaichao.xyz/Out-of-SightTrajPred/assets/arch.png">
</a>
</div>

---

## Datasets:

### Processed Vi-Fi and JRDB dataset [(Link to Google Drive)](https://drive.google.com/drive/folders/1W6ze1z8X54kK9BOgYbXYQj_AScf79Z-q?usp=sharing)
Please download pkl files for preprocessed datasets, and put them in "JRDB/jrdb.pkl" and "vifi_dataset_gps/vifi_data.pkl".
If need, you can also access original data from [Vi-Fi Dataset](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home) or [JRDB](https://jrdb.erc.monash.edu/)

# Training

##
    python train.py train --model VisionPosition --gpus 0 --phase 2 --dataset vifi --dec_model Transformer  --learning_rate 0.001
The training and print accuracy logs are located under ./checkpoints/ 

---

## BibTeX
    @inproceedings{Zhang2024OOSTraj,
        title={OOSTraj: Out-of-Sight Trajectory Prediction With Vision-Positioning Denoising},
        author={Haichao Zhang, Yi Xu, Hongsheng Lu, Takayuki Shimizu, and Yun Fu},
        booktitle={In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2024}
    }

## License
The majority of OOSTraj is licensed under an [Apache License 2.0](https://github.com/ma-xu/Rewrite-the-Stars/blob/main/LICENSE)
