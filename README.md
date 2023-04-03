# MonoViT

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **MonoViT: Self-Supervised Monocular Depth Estimation with a Vision Transformer** [arxiv](https://arxiv.org/abs/2208.03543)
> 
>Chaoqiang Zhao*, Youmin Zhang*, Matteo Poggi, Fabio Tosi, Xianda Guo,Zheng Zhu, Guan Huang, Yang Tang, Stefano Mattoccia

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/monovit-self-supervised-monocular-depth/monocular-depth-estimation-on-kitti-eigen-1)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1?p=monovit-self-supervised-monocular-depth)

<div class='paper-box'><div class='paper-box-image'><img src='fig/kittiandds.png' alt="sym" width="90%"></div>
<div class='paper-box-text' markdown="1">


If you find our work useful in your research please consider citing our paper:

```
@inproceedings{monovit,
  title={MonoViT: Self-Supervised Monocular Depth Estimation with a Vision Transformer},
  author={Zhao, Chaoqiang and Zhang, Youmin and Poggi, Matteo and Tosi, Fabio and Guo, Xianda and Zhu, Zheng and Huang, Guan and Tang, Yang and Mattoccia, Stefano},
  booktitle={International Conference on 3D Vision},
  year={2022}
}
```



## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0
pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8
pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image
pip3 install mmcv-full==1.3.0 mmsegmentation==0.11.0  
pip install timm einops IPython
```
We ran our experiments with PyTorch 1.9.0, CUDA 11.1, Python 3.7 and Ubuntu 18.04.

Note that our code is built based on [Monodepth2](https://github.com/nianticlabs/monodepth2)

## Results on KITTI

We provide the following  options for `--model_name`:

| `--model_name`          | Training modality | Pretrained? | Model resolution  |Abs Rel| Sq Rel| RMSE| RMSE log|  delta < 1.25  | delta < 1.25^2  | delta < 1.25^3  |
|-----------------------|-------------|------|-----------------|----|----|----|------|--------|--------|--------|
| [`mono_640x192`](https://drive.google.com/drive/folders/1VWDPuqiMPDD2P--Oka-yJgh8z7ouCX4D?usp=sharing)          | Mono              | Yes | 640 x 192                | 0.099 |0.708 |4.372| 0.175 |0.900 |0.967| 0.984|
| [`mono+stereo_640x192`](https://drive.google.com/drive/folders/1_HPsL1Vg3s0LdOykfTT0aMlE6-u3IxQn?usp=sharing)   | Mono + Stereo     | Yes | 640 x 192                | 0.098| 0.683| 4.333| 0.174| 0.904| 0.967| 0.984|
| [`mono_1024x320`](https://drive.google.com/drive/folders/1EDTSZ59CGW9rUoDL3EwEKn3PpZpUUGsS?usp=sharing)         | Mono              | Yes | 1024 x 320               | 0.096|  0.714|  4.292|  0.172|  0.908|  0.968|  0.984|
| [`mono+stereo_1024x320`](https://drive.google.com/drive/folders/1tez1RQFO33MMyVAq_gkOVHoL2TO98-TH?usp=sharing)  | Mono + Stereo     | Yes | 1024 x 320               | 0.093 |0.671 |4.202 |0.169 |0.912 |0.969 |0.985|
| [`mono_1280x384`](https://drive.google.com/drive/folders/1l3egRvLaoBqgYrgfktgpJt613QwZ4twT?usp=sharing)         | Mono              | Yes | 1280 x 384               | 0.094 |0.682| 4.200| 0.170| 0.912| 0.969| 0.984|

## Robustness

| Model | Modality | mCE (%) | mRR (%) | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| [MonoDepth2<sub>R18</sub>]()| Mono | 100.00 | 84.46 |  0.119 | 0.130     | 0.280     | 0.155     | 0.277     | 0.511     | 0.187     | 0.244     | 0.242     | 0.216     | 0.201     | 0.129     | 0.193     | 0.384     | 0.389     | 0.340     | 0.388     | 0.145     | 0.196     |
| [MonoDepth2<sub>R18+nopt</sub>]() | Mono | 119.75 | 82.50 | 0.144 | 0.183     | 0.343     | 0.311     | 0.312     | 0.399     | 0.416     | 0.254     | 0.232     | 0.199     | 0.207     | 0.148     | 0.212     | 0.441     | 0.452     | 0.402     | 0.453     | 0.153     | 0.171     |
| [MonoDepth2<sub>R18+HR</sub>]() | Mono | 106.06 | 82.44 | 0.114 | 0.129     | 0.376     | 0.155     | 0.271     | 0.582     | 0.214     | 0.393     | 0.257     | 0.230     | 0.232     | 0.123     | 0.215     | 0.326     | 0.352     | 0.317     | 0.344     | 0.138     | 0.198     |
| [MonoDepth2<sub>R50</sub>]() | Mono | 113.43 | 80.59 | 0.117 | 0.127 | 0.294 | 0.155 | 0.287 | 0.492 | 0.233 | 0.427 | 0.392 | 0.277 | 0.208 | 0.130 | 0.198 | 0.409 | 0.403 | 0.368 | 0.425 | 0.155 | 0.211 |
| [MaskOcc]() | Mono | 104.05 | 82.97 | 0.117 | 0.130 | 0.285 | 0.154 | 0.283 | 0.492 | 0.200 | 0.318 | 0.295 | 0.228 | 0.201 | 0.129 | 0.184 | 0.403 | 0.410 | 0.364 | 0.417 | 0.143 | 0.177 |
| [DNet<sub>R18</sub>]() | Mono | 104.71 | 83.34 | 0.118 | 0.128 | 0.264 | 0.156 | 0.317 | 0.504 | 0.209 | 0.348 | 0.320 | 0.242 | 0.215 | 0.131 | 0.189 | 0.362 | 0.366 | 0.326 | 0.357 | 0.145 | 0.190 |
| [CADepth]() | Mono | 110.11 | 80.07 | 0.108 | 0.121 | 0.300 | 0.142 | 0.324 | 0.529 | 0.193 | 0.356 | 0.347 | 0.285 | 0.208 | 0.121 | 0.192 | 0.423 | 0.433 | 0.383 | 0.448 | 0.144 | 0.195 |
| [HR-Depth]() | Mono | 103.73 | 82.93 | 0.112 | 0.121 | 0.289 | 0.151 | 0.279 | 0.481 | 0.213 | 0.356 | 0.300 | 0.263 | 0.224 | 0.124 | 0.187 | 0.363 | 0.373 | 0.336 | 0.374 | 0.135 | 0.176 |
| [DIFFNet<sub>HRNet</sub>]() | Mono | 94.96 | 85.41 | 0.102 | 0.111 | 0.222 | 0.131 | 0.199 | 0.352 | 0.161 | 0.513 | 0.330 | 0.280 | 0.197 | 0.114 | 0.165 | 0.292 | 0.266 | 0.255 | 0.270 | 0.135 | 0.202 |
| [ManyDepth<sub>single</sub>]() | Mono | 105.41 | 83.11 | 0.123 | 0.135 | 0.274 | 0.169 | 0.288 | 0.479 | 0.227 | 0.254 | 0.279 | 0.211 | 0.194     | 0.134     | 0.189 | 0.430 | 0.450 | 0.387 | 0.452 | 0.147 | 0.182 |
| [FSRE-Depth]() | Mono | 99.05 | 83.86 | 0.109 | 0.128 | 0.261 | 0.139 | 0.237 | 0.393 | 0.170 | 0.291 | 0.273 | 0.214 | 0.185 | 0.119 | 0.179 | 0.400 | 0.414 | 0.370 | 0.407 | 0.147 | 0.224 |
| [MonoViT<sub>MPViT</sub>]() | Mono | 79.33 | 89.15 | 0.099 | 0.106 | 0.243 | 0.116 | 0.213 | 0.275 | 0.119 | 0.180 | 0.204 | 0.163 | 0.179 | 0.118 | 0.146 | 0.310 | 0.293 | 0.271 | 0.290 | 0.162 | 0.154 |
| [MonoViT<sub>MPViT+HR</sub>]() | Mono | 70.79 | 90.67 | 0.090 | 0.097 | 0.221 | 0.113 | 0.217 | 0.253 | 0.113 | 0.146 | 0.159 | 0.144 | 0.175 | 0.098 | 0.138 | 0.267 | 0.246 | 0.236 | 0.246 | 0.135 | 0.145 |

The [RoboDepth Challenge Team](https://github.com/ldkong1205/RoboDepth) is evaluating the robustness of different depth estimation algorithms. MonoViT has achieved the outstanding robustness.

## 💾 KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

The above conversion command creates images which match our experiments, where KITTI `.png` images were converted to `.jpg` on Ubuntu 16.04 with default chroma subsampling `2x2,1x1,1x1`.
We found that Ubuntu 18.04 defaults to `2x2,2x2,2x2`, which gives different results, hence the explicit parameter in the conversion command.

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.


## ⏳ Training

PLease download the ImageNet-1K pretrained MPViT [model](https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth) to `./ckpt/`.

For training, please download monodepth2, replace the depth network, and revise the setting of the depth network, the optimizer and learning rate according to `trainer.py`. 

Because of the different torch version between MonoViT and Monodepth2, the func `transforms.ColorJitter.get_params` in dataloader should also be revised to `transforms.ColorJitter`.

By default models and tensorboard event files are saved to `./tmp/<model_name>`.
This can be changed with the `--log_dir` flag.


**Monocular training:**
```shell
python train.py --model_name mono_model --learning_rate 5e-5
```

**Monocular + stereo training:**
```shell
python train.py --model_name mono+stereo_model --use_stereo --learning_rate 5e-5
```


### GPUs

The code of the Single GPU version can only be run on a single GPU.
You can specify which GPU to use with the `CUDA_VISIBLE_DEVICES` environment variable:
```shell
CUDA_VISIBLE_DEVICES=1 python train.py --model_name mono_model
```

## 📊 KITTI evaluation

To prepare the ground truth depth maps, please follow the monodepth2.

...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model` (Note that please use `evaluate_depth.py` for 640x192 models and `evaluate_hr_depth.py --height 320/384 --width 1024/1280` for the others):
```shell
python evaluate_depth.py --load_weights_folder ./tmp/mono_model/models/weights_19/ --eval_mono
```


An additional parameter `--eval_split` can be set.
The three different values possible for `eval_split` are explained here:

| `--eval_split`        | Test set size | For models trained with... | Description  |
|-----------------------|---------------|----------------------------|--------------|
| **`eigen`**           | 697           | `--split eigen_zhou` (default) or `--split eigen_full` | The standard Eigen test files |
| **`eigen_benchmark`** | 652           | `--split eigen_zhou` (default) or `--split eigen_full`  | Evaluate with the improved ground truth from the [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) |
| **`benchmark`**       | 500           | `--split benchmark`        | The [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) test files. |

## Contact us

Contact us: zhaocqilc@gmail.com

## Acknowledgement
Thanks the authors for their works:

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[MPVIT](https://github.com/youngwanLEE/MPViT)

[HR-Depth](https://github.com/shawLyu/HR-Depth)

[DIFFNet](https://github.com/brandleyzhou/DIFFNet)
