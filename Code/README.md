# Code for the whole model
## File description
This folder contains 5 python files and 1 readme file
* GMou_Whole.py:    code to start with, main function locates in.
* model.py:         where the actual model locates in.
* data.py:          reading data and dataset object for data loaders.
* mask.py:          simple code for many kinds of masking.
* printHyper.py:    simple code for showing best hyper-parameters.

## Environment and libraries
Code written in python3, preferrably > python3.5

Tested under 1 1080Ti GPU, 64GB RAM memory (you do not need that much memory but be aware that I have so many instances and all embeddings are huge)

Following libraries are needed to install
```
torch
matplotlib
pprint
sklearn
pandas
```

## How to play with it:
1. Make sure we have generated corresponding data files under ../Data/. We provided sample data but these are very small (only 2 batch for each split), only showing that the code is workable. The original raw data are all publically available. One should be able to download and preprocess them. If you are interested please refer to our paper, we cited all those sources.
2. Say we have 1 gpu with index 0, then execute the following command in your terminal
```cmd
CUDA_VISIBLE_DEVICES=0 nohup python3 -u GMou_Whole.py > result.out &
```
When everything finished, a general report will be printed in result.out. Extra npy files are generated for further analysis/comparison. A pdf file is plotted for learning rate tuning.

3. You can then simply run the following command to check the best hyper-parameters so far.
```cmd
python3 printHyper.py
```
4. If you wish to tune the model on your own dataset. Then you may wish to change hypers in Gmou_Whole.py line 310 to line 336. As I have many cpu cores I set data loader workers as 8, feel free to change them to smaller numbers for your own case.

5. So far I have tried very hard in writing clear documentations. Most things are straightforward if you fully read the code through. I know as always, devil lives in details, plus, there is never perfect code and mine is far from it. So if there's still sth unclear to you, feel free to contact me. gmou@wpi.edu

