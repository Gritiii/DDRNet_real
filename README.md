# DDRNet_real
# DDRNet

## Installation
To set up the environment, please follow these steps:

1. Create and activate a conda environment :
    ```shell
    conda create -n DDR python=3.8
    conda activate DDR
    ```


2. Install the required packages:
    ```shell
    pip install -r requirements.txt
    ```


## Datasets

You can download the pre-processed datasets from the following  link , and extraction code is 0000:

Vaihingen: https://pan.baidu.com/s/1wVgiNohIopu0evn871PDqQ

LoveDA: https://pan.baidu.com/s/1DyrDvJmKtf_jJGqCCJ3dSA

Potsdam: https://pan.baidu.com/s/1WxNuyERdCLaE1zqfnIlPtw


After downloading and extracting, place the datasets in the `./Datasets` directory.



## Project Structure

The folder structure for the `DDRNet` directory is as follows:


```plaintext
DDRNet/
├── config
│   ├── loveda
│   │   ├── ddrnet.py
│   ├── vaihingen
│   │   ├── ddrnet.py
│   ├── potsdam
│   │   ├── ddrnet.py
├── ddrnet
│   ├── datasets
│   ├── losses
│   ├── models
├── pretrain_weights
│   ├── swsl_resnet18.pth
│   ├── stseg_tiny.pth
│   ├── swsl_resnet50.pth
│   ├── vgg16.pth
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA)
├── fig_results (save the masks predicted by models)
├── datasets
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   ├── Rural
│   │   ├── Val 
│   │   │   ├── Urban
│   │   │   ├── Rural
│   │   ├── Test
│   ├── vaihingen
│   │   ├── train
│   │   ├── test
│   ├── potsdam
│   │   ├── train
│   │   ├── test
├── requirements.txt
├── train.py
├── vaihingen_test.py
├── loveda_test.py
├── potsdam_test.py
├── inference_huge_image.py
```
## Pretrained Weights of Backbones

[Baidu Disk](https://pan.baidu.com/s/1kfW8vvAhCvbGK81vQQczoA) : extraction code is：0000


## Training

"-c" means the path of the config, use different **config** to train different models.

```
python train_supervision.py -c config/loveda/ddrnet.py
```
```
python train_supervision.py -c config/potsdam/ddrnet.py
```
```
python train_supervision.py -c config/vaihingen/ddrnet.py
```
## Testing

**Vaihingen**
```
python vaihingen_test.py -c config/vaihingen/ddrnet.py -o fig_results/vaihingen/ddrnet --rgb -t 'd4'
```

**Potsdam**
```
python potsdam_test.py -c config/potsdam/ddrnet.py -o fig_results/potsdam/ddrnet --rgb -t 'd4'
```

**LoveDA** 
```
python loveda_test.py -c config/loveda/dcswin.py -o fig_results/loveda/ddrnet -t 'd4' --val
```

## Inference on huge remote sensing image
```
python inference_huge_image.py \
-i datastes/vaihingen/test_images \
-c config/vaihingen/ddrnet.py \
-o fig_results/vaihingen/ddrnet_huge \
-t 'd4' -ph 512 -pw 512 -b 2 -d "pv"
```
