# Patch based CNN using Scattering transform

## Requirements
```
pytorch <= 1.7
numpy
```
Install [multi-gpu kymatio with oversampling](https://github.com/louity/kymatio/tree/multigpu)

You also need to download the ImageNet 2012 dataset.

## BagnetScattering

Scattering transform `J=4` + 1layer 3x3 convolution + N-1 layers of 1x1 convolutions + global average pooling + linear fc.
  * order 2 + 8 layers of width 2048 + oversampling
    ```
      python main.py -a bagnetscattering --scattering-J 4 --scattering-order2 --layer-width 2048 --n-iterations 8 -j 8 --epochs 100 --scattering-oversampling 1  <PATH_TO_IMAGENET>
    ```
    84.5 % top5 test accuracy

  * order 1 + 8 layers of width 2048 + oversampling
    ```
      python main.py -a bagnetscattering --scattering-J 4 --layer-width 2048 --n-iterations 8 -j 8 --epochs 100 --scattering-oversampling 1  <PATH_TO_IMAGENET>
    ```
    82.2 % top5 test accuracy
  * order 1 + 8 layers of width 2048 + no oversampling
    ```
      python main.py -a bagnetscattering --scattering-J 4 --layer-width 2048 --n-iterations 8 -j 8 --epochs 100 <PATH_TO_IMAGENET>
    ```
    79.5 % top5 test accuracy
  * order 1 + 4 layers of width 2048 + oversampling
    ```
      python main.py -a bagnetscattering --scattering-J 4 --layer-width 2048 --n-iterations 4 -j 8 --epochs 100 --scattering-oversampling 1  <PATH_TO_IMAGENET>
    ```
    79.2 % top5 test accuracy
  * order 1 + 8 layers of width 1024 + oversampling
    ```
      python main.py -a bagnetscattering --scattering-J 4 --layer-width 1024 --n-iterations 4 -j 8 --epochs 100 --scattering-oversampling 1  <PATH_TO_IMAGENET>
    ```
    80.5 % top5 test accuracy
