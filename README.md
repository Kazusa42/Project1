# Project1
Code for thesis. Mainly use YOLO v4 to do detection.

This project focus on backbone structure, to see how it influence the performance of a model. At every begining, ResNet50, CSPDarkNet53, DenseNet, MobileNet were tried. However, they are not novle enough.

Then ConvNext has been applied as a new approach.

Now, the model is training under a `MHSA` block.

---

## AttentionNeck  
Use a pure global mulit-head self attention block to replace the depth-wise convolution in bottleneck. More etails about this design is shown blow.


---

## Code Structure
1. `./nets` contains several backbone structures.  
2. If want to add more backbone structure(s), please add under `./nets`.  
3. After add backbone, please re-write a _class_ method about new backbone in `./nets/backbones.py`. And add your backbone in `BACKBONE_LIST` in `configure.py`.  
4. Other part is complete enough. Try not to change them.

---

## Train
1. Change training schedule in `configure.py`.  
2. Make sure `CLASSES_PATH` and `DATASET_PATH` and `MODEL_PATH` are matched.
3. Weights file will be stored every epoch. It's okey to interrupt and re-start.
4. When re-start, please change `MODEL_PATH` to the weights file when interrupting, as well as `INIT_EPOCH`.  

---

## Evaluate
1. Change the `MODEL_PATH` to the weight which needs to be evaluate.  
2. Make sure you have already got `test.txt` under dir `./your_dataset_name/ImageSets/Main`.  
3. Run `get_map.py`. Ppay attention to papram `MINOVERLAP`.  

---

## Dataset
The dataset is DOTA v1.0. The original train set will be used as train and val set. The original val set will be treated as test set.

For training, the original train set is cropped into 2 scales:  
```
subsize = 640, overlap = 50
subsize = 1280, overlap = 100
```

For evaluate, how to crop the original val dataset has not been decided yet.

About how to processing the data, refer to another project `DOTA-processing`.

