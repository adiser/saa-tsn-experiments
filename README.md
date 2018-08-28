# **A*STAR I2R Situational Awareness Analytics Department - Temporal Segment Networks Experiments Codebase**
_by Sergi Adipraja Widjaja -- Experiments conducted in May - August 2018_

# Overview
This codebase is to replicate the experiments that I conducted during my summer attachment at A*STAR Institute for Infocomm Research. This work revolves around the famous [Temporal Segment Networks for Action Recognition][temporal-segment-networks] framework [(default PyTorch implementation)][tsn-pytorch]

In this documentation, I will go through some of the extended features that I implemented in my codebase during my attachment.
* [Dataset Preparation](#dataset-preparation)
* [TSN-PyTorch Preparation](tsn-pytorch-preparation)
* [OpenPose Heatmap Generation](#openpose-heatmap-generation)
* [Windowed Rank Pooling Implementation](#windowed-rank-pooling)
 
# Dataset Preparation
We will be using the [original codebase][temporal-segment-networks] for the Temporal Segment Networks for frame extraction and video file list generation

Do the following things:
### 1. Download the HMDB51 dataset and unrar it
```
$ mkdir data; cd data
$ wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
$ unrar x hmdb51-org.rar rars/
$ for a in $(ls rars); do unrar x "rars/${a}" videos/; done;
```
### 2. Clone and build the temporal-segment-networks library
```
$ cd ..
$ mkdir utils
$ git clone --recursive https://www.github.com/yjxiong-temporal-segment-networks utils/temporal-segment-networks
$ cd utils/temporal-segment-networks
$ bash build_all.sh
```
If error persists while building OpenCV, do take note of the following pointers:  
* Make sure that you are using CUDA-8.0
* Change the OpenCV version variable inside the _build_all.sh_ script to be 2.4.13.6  
  
If error still persists, feel free to contact me at swidjaja001@e.ntu.edu.sg

### 3. Generate the frames
```
$ bash scripts/extract_optical_flow.sh ../../data/hmdb51/ ../../data/hmdb51_frames/ <NUM_GPU>
```

And the dataset is ready! In this repo, I have provided training and validation file lists as an interface between the training and testing with the dataset itself (placed under /data)


# TSN-PyTorch Preparation
This part is a tad bit peculiar as we are going to install [PyTorch version 0.1.12][torch-0.1.12]. The reason why is that in the next step(s), we are going to leverage on the [PyTorch pretrained models repo](https://www.github.com/cadene/pretrainedmodels). This is crucial as the models that we are going to work with are Inceptionv3 and BNInception in which both do not ship in PyTorch 0.2.0 or newer (at the time when this documentation is written).

Do the following things:
### 1. Create a pytorch virtual environment
This is my preferred way to go as using alternative methods such as using conda will potentially confuse your environment

```
$ mkdir envs; cd envs
$ python3 -m venv pytorch-0.1.12
$ cd pytorch-0.1.12
$ source bin/activate

# The PyTorch wheel file that you download should be compatible with whatever your python version is
$ pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
$ pip install numpy
$ pip install torchvision
```

### 2. Cloning my version of tsn-pytorch
The next step is to use my version of tsn-pytorch to train and test models. At this point, you can use the default tsn-pytorch developed by @yjxiong repo. But for the next steps (Rank Pooling and Pose Heatmaps Implementation) you will need to use my version of the repo as it has several command line features that is scalable for other usages

```
# Go to the root directory
$ cd ../../
# Clone my version of the repository
$ git clone https://github.com/adiser/tsn-pytorch
```
### 3. Training and testing
Example training command:
```
$ cd ../../tsn-pytorch
$ python main.py hmdb51 RGB data/hmdb51_rgb_train_split_1.txt data/hmdb51_rgb_val_split_1.txt --arch BNInception --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b  64 -j 8 --dropout 0.8 --snapshot_pref hmdb51_bninception_1
```

Example testing command:
```
$ python test_models.py hmdb51 RGB data/hmdb51_rgb_val_split_1.txt <CHECKPOINT FILE> --arch BNInception --save_scores ../scores/sample_score_file.npz

```

Examine score file
```
$ cd utils/temporal-segment-networks
$ python tools/eval_scores.py <SCORE_FILES> --score_weights <WEIGHTAGES> 
```

# OpenPose Heatmap Generation
Clone [my branch][openpose_branch] of [openpose][openpose]
```
$ cd utils
$ git clone https://www.github.com/adiser/openpose 
```
Go to the directory user_code and execute main.py to generate pose heatmaps from your video frames
```
$ cd openpose/examples/user_code
$ python main.py
```


# Windowed Rank Pooling
First, you will need to clone [my branch][my_branch] of the famous code repo [dynamic-image-nets][dynamic-image-nets]
```
$ cd utils
$ git clone https://www.github.com/adiser/dynamic-image-nets
$ cd dynamic-image-nets

```
Run the following command in MATLAB to apply windowed rank pooling for one video folder
```
compute_windowed_dynamic_images('../../data/hmdb51_frames', 'video_frames_k', 1, 10)
```
Run the following command in MATLAB to apply windowed rank pooling for all of the frame folders
```
save_from_folders('../../data/hmdb51_frames')
```

# Overview of tsn-pytorch extended features
I extended several features from the original codebase so that the code can read different modalities of input indicated by the image file prefix

### 1. Read different input modalities
For example if you have generated a dataset with this particular structure
```
data/hmdb51_frames/
    ├── video_frames_k
    │   ├── example_custom_prefix_00001.jpg
    │   ├── example_custom_prefix_00002.jpg
    │   ├── example_custom_prefix_00003.jpg
    │   ├── example_custom_prefix_00004.jpg
    │   ├── example_custom_prefix_00005.jpg
```

You can make your code to be train from the custom inputs you have generated (e.g. through some experiments) by doing:
```
$ python main.py ... --custom_prefix some_custom_prefix_
$ python test.py ... --custon_prefix some_custom_prefix_
```

At this point of the process, your dataset structure should look something like this:
```
data/hmdb51_frames/
    ├── video_frames_k
    │   ├── img_00001.jpg
    │   ├── img_00002.jpg
    │   ├── img_00003.jpg
    │   ├── ...
    │   ├── rprgb_00001.jpg
    │   ├── rprgb_00002.jpg
    │   ├── rprgb_00003.jpg
    │   ├── ...
    │   ├── hmpaf_00001.jpg
    │   ├── hmpaf_00002.jpg
    │   ├── hmpaf_00003.jpg
```
Where frames with prefix 'rprgb_' are the rank pooled RGB images and frames with prefix 'hmpaf_' are the extracted pose heatmaps complete with PAF (Part Affinity Fields)

You can use the added feature discussed above by doing something like:
```
$ python main.py hmdb51 RGB data/hmdb51_rgb_train_split_1.txt data/hmdb51_rgb_val_split_1.txt --arch BNInception --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b  64 -j 8 --dropout 0.8 --snapshot_pref hmdb51_bninception_1 --custom_prefix hmpaf_
$ python main.py hmdb51 RGB data/hmdb51_rgb_train_split_1.txt data/hmdb51_rgb_val_split_1.txt --arch BNInception --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b  64 -j 8 --dropout 0.8 --snapshot_pref hmdb51_bninception_1 --custom_prefix rprgb_
```
_Note the custom_prefix argument I added_

### 2. Use models that are pretrained with kinetics
By default, the models used in the experiments are models that were pretrained using the imagenet dataset. For video recognition purposes, it has been proven extensively that pretraining with kinetics improves the model accuracy greatly. In this expreriment, we can use the models that are pretrained with kinetics as a starting point by adding an extra argument
```
$ python main.py ... --pretrained_with_kinetics 1
```
The integer input to the argument will dictate which model we are going to use as a starting point. Integer input of 1 will import the RGB pretrained model, while integer input of 2 will import the optical flow pretrained models. Note that the available pretrained models are BNInception and InceptionV3.



[temporal-segment-networks]: https://www.github.com/yjxiong/temporal-segment-networks
[tsn-pytorch]: https://www.github.com/yjxiong/tsn-pytorch
[torch-0.1.12]: http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl
[openpose_branch]: https://www.github.com/adiser/openpose
[openpose]: https://www.github.com/CMU-Perceptual-Computing-Lab/openpose
[my_branch]: https://www.github.com/adiser/dynamic-image-nets
[dynamic-image-nets]: https://www.github.com/hbilen/dynamic-image-nets