# PointMoment
# Note: If your work uses this algorithm or makes improvements based on it, please be sure to cite this paper. Thank you for your cooperation.

# 注意：如果您的工作用到了本算法，或者基于本算法进行了改进，请您务必引用本论文，谢谢配合



# PointMoment: Mixed-Moment-based Self-Supervised Representation Learning for 3D Point Clouds

*Xin Cao* <sup>1,2</sup>, *Xinxin Han*<sup>1</sup>, *Haoyu Wang*<sup>1</sup>,*Qiuquan Zhu*<sup>1</sup>,*Ping Zhou*<sup>1,2</sup>,*Kang Li*<sup>1,2</sup>

1.School of Information Science and Technology, Northwest University, Xi’an, Shaanxi 710127, China

2.National and Local Joint Engineering Research Center for cultural Heritage Digitization, Xi’an, Shaanxi 710127, China
This repository is the official implementation of PointMoment: Mixed-Moment-based Self-Supervised Representation Learning for 3D Point Clouds, Scientific Reports. 2024.

Please feel free to reach out for any questions or discussions!


## Requirements:

Make sure the following environments are installed.

```
python=3.8.13
pytorch==1.12.1
torchvision=0.13.1
cudatoolkit=11.3.1
matplotlib=3.5.2
tqdm=4.64.1
scipy=1.7.1
wandb=0.12.1
seaborn=0.11.2
```

## Datasets 

We use ModelNet40 provided by [Princeton](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) for pretrain.

Run the command below to download all the datasets (ModelNet40, ScanObjectNN, ShapeNetPart) to reproduce the results.

```bash
cd datasets
source download_data.sh
```


## Training

```bash
# Pre-train a projector
python train_pointmoment.py
```


## Testing

```bash
# Test classification result
python train_pointmomeng.py 

# Test segmentation result
python eval_home_seg.py
```

## Citation

```
@article{cao2023pointmoment,
  title={PointMoment: mixed-moment-based self-supervised representation learning for 3D point clouds},
  author={Cao, Xin and Han, Xinxin and Wang, Haoyu and Zhu, Qiuquan and Zhou, Ping and Li, Kang}
}
```

## Acknowledgements

We would like to thank and acknowledge referenced codes from the following repositories:

https://github.com/WangYueFt/dgcnn

https://github.com/charlesq34/pointnet

https://github.com/AnTao97/dgcnn.pytorch
