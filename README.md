# Slimmable Multi-task Image Compression for Human and Machine Vision
![image](https://github.com/May-Yao/slimmable_image_compression/blob/main/framework_02.png)
Jiangzhong Cao; Ximei Yao; Huan Zhang; Jian Jin; Yun Zhang; Bingo Wing-Kuen Ling  
Slimmable Multi-Task Image Compression for Human and Machine Vision  
[[paper](https://ieeexplore.ieee.org/document/10080941)]
# Abstract
In the Internet of Things (IoT) communications, visual data is frequently processed among intelligent devices using artificial intelligence algorithms, replacing humans for analyzing and
decision-making while only occasionally requiring human’s scrutiny. However, due to high redundancy of compressive encoders, existing image coding solutions for machine vision are not efficient at runtime. To balance the rate-accuracy performance and efficiency of image compression for machine vision while attaining high-quality reconstructed images for human vision, this paper introduces a novel slimmable multi-task compression framework for human and machine vision in visual IoT applications. Firstly, the image compression for human and machine vision under the constraint of bandwidth, latency, computational resources are modelled as a multi-task optimization problem. Secondly, slimmable encoders are employed to multiple human and machine vision tasks in which the parameters of the sub-encoder for machine vision tasks are shared among all tasks and jointly learned. Thirdly, to solve the feature match between latent representation and intermediate features of deep vision networks, feature transformation networks are introduced as decoders of machine vision feature compression. Finally, the proposed framework is successfully applied to human and machine vision tasks’ scenarios, e.g., object detection and image reconstruction. Experimental results show that the proposed method outperforms baselines and other image compression approaches on machine vision tasks with higher efficiency (shorter latency) in two vision tasks’ scenarios while retaining comparable quality on image reconstruction.  
# Installation
The codebases are built on top of [CompressAI](https://github.com/InterDigitalInc/CompressAI) , [Slimmabe Networks](https://github.com/JiahuiYu/slimmable_networks) and [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3).
### Requirements
- Linux or macOS with Python ≥ 3.7    
- PyTorch ≥ 1.7 and torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this   
### Steps
1. Install and build libs  
(1) Install yolov3 according to [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and modifiy the file PyTorchYOLOv3/pytorchyolo/model.py    
(2) Install CompressAI according to [CompressAI](https://github.com/InterDigitalInc/CompressAI)  
(3) Install this project  
```
git clone https://github.com/May-Yao/slimmable_image_compression.git
cd slimmable_image_compression
```  


2. Dataset preparation 
You can download COCO2014 dataset from [HERE](https://cocodataset.org/#download) and annotated it according to [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

3. Train  
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29303 train_hyper_DDP.py --checkpoint checkpoint.pth.tar
```
Training algorithms:  
![image](https://github.com/May-Yao/slimmable_image_compression/blob/main/Training_algorithms.png)

4. Results  
Object detection:  
![image](https://github.com/May-Yao/slimmable_image_compression/blob/main/Object_detection.png)

Image reconstruction:  
![image](https://github.com/May-Yao/slimmable_image_compression/blob/main/Image_reconstruction.png) 

# Citing
If you use our work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:  
@ARTICLE{10080941,  
  author={Cao, Jiangzhong and Yao, Ximei and Zhang, Huan and Jin, Jian and Zhang, Yun and Ling, Bingo Wing-Kuen},  
  journal={IEEE Access},   
  title={Slimmable Multi-Task Image Compression for Human and Machine Vision},   
  year={2023},  
  volume={11},  
  pages={29946-29958},  
  doi={10.1109/ACCESS.2023.3261668}}  
