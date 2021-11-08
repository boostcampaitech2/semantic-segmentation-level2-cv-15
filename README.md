# Semantic Segmentation (재활용 쓰레기 분류)

## Table of Contents

- [Background](#background)
- [Usage](#usage)
  - [Structure](#Structure)
  - [Requirements](#install)
  - [Getting_Started](#Getting_Started)
- [Result](#Result)
- [Reference](#Reference)

## Background

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## Usage
### Structure
```sh
|-- Custom_Baseline_Code # baseline code for team
|   |-- deeplab_xception_model
|   |-- main.py
|   |-- myaugmentations.py
|   |-- myconfig.py
|   |-- mydataset.py
|   |-- myloss.py
|   |-- mymodel.py
|   |-- myoptimizer.py
|   |-- mytrain.py
|   |-- myvalidation.py
|   |-- requirements.txt
|   |-- utils.py
|-- EDA # EDA ipynb files
|   |-- augmentation_test.ipynb
|   |-- avg_check_eda.ipynb
|   |-- seg_test.ipynb
|   |-- test_output.ipynb
|   |-- train_output.ipynb
|   `-- utils.py
|-- etc
|   `-- smp_practice.ipynb # guide for SMP library
|-- mmsegmentation
|   |-- configs # config folder for mmsegmentation
|   |-- dataset
|   |-- mmseg # mmsegmentation core
|   `-- tools
`-- utils
    |-- SWA.py
    |-- Stratified k-Fold.ipynb
    |-- ensamble_and_crf__viz.ipynb
    |-- ensemble_all.ipynb
    |-- mmseg_ensemble.ipynb
    `-- mmseg_inference.ipynb
```
### Install
- Requirements
  ```sh
  $ pip install -r requirements.txt
  ```
- MMSegmentation
  ```sh
  $ conda create -n open-mmlab python=3.7 -y
  $ conda activate open-mmlab
  $ conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch

  $ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

  $ git clone https://github.com/open-mmlab/mmsegmentation.git

  $ cd mmsegmentation
  $ pip install -e .  # or "python setup.py develop"
  ```
- SMP(Segmentation Models Pytorch)
  ```sh
  $ pip install git+https://github.com/qubvel/segmentation_models.pytorch
  ```

### Getting_Started
1. Custom_Baseline_Code
    ```sh
    $ cd Custom_Baseline_Code
    $ python main.py

    [Parameter setting : myconfig.py]
    ```
2. MMSegmentation
    ```sh
    $ cd mmsegmentation
    $ python ./tools/train.py [Config File]
      [ex) python ./tools/train.py ./config/1.myconfig/upernet_swin.py]
    ```

## Result

<img src="https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-15/blob/master/Custom_Baseline_Code/image/image0.GIF" width="600px" height="400px"></img><br/>

<img src="https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-15/blob/master/Custom_Baseline_Code/image/image1.GIF" width="800px" height="400px"></img><br/>


## Reference

<details>
  <summary>MMSegmentation</summary>

- MMSegmentation
  ```latex
  @misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
    }
  ```

- UPerNet
  ```latex
  @inproceedings{xiao2018unified,
  title={Unified perceptual parsing for scene understanding},
  author={Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={418--434},
  year={2018}
  }
  ```

- OCRNet
  ```latex
  @article{YuanW18,
    title={Ocnet: Object context network for scene parsing},
    author={Yuhui Yuan and Jingdong Wang},
    booktitle={arXiv preprint arXiv:1809.00916},
    year={2018}
  }

  @article{YuanCW20,
    title={Object-Contextual Representations for Semantic Segmentation},
    author={Yuhui Yuan and Xilin Chen and Jingdong Wang},
    booktitle={ECCV},
    year={2020}
  }
  ```

- DeepLabV3+
  ```latex
  @inproceedings{deeplabv3plus2018,
    title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
    author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
    booktitle={ECCV},
    year={2018}
  }
  ```

- Swin Transformer
  ```latex
  @article{liu2021Swin,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={arXiv preprint arXiv:2103.14030},
    year={2021}
  }
  ```

- HRNet
  ```latex
  @inproceedings{SunXLW19,
    title={Deep High-Resolution Representation Learning for Human Pose Estimation},
    author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
    booktitle={CVPR},
    year={2019}
  }
  ```
</details>

<details>
  <summary>SMP</summary>
  
  - SMP
    ```latex
    @misc{Yakubovskiy:2019,
      Author = {Pavel Yakubovskiy},
      Title = {Segmentation Models Pytorch},
      Year = {2020},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
    }
    ```

  - EfficientNet
    ```latex
      @misc{tan2020efficientnet,
        title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
        author={Mingxing Tan and Quoc V. Le},
        year={2020},
        eprint={1905.11946},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
        }
    ```

    ```bash
    git clone https://github.com/lukemelas/EfficientNet-PyTorch
    ```
</details>


