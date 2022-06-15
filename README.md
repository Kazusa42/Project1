# Project1
Code for thesis. Mainly use YOLO v4 to do detection.

This project focus on backbone structure, to see how it influence the performance of a model. At every begining, I tried ResNet50, CSPDarkNet53, DenseNet, MobileNet.
However, they are not novle enough.

So, the second step is to apply [__ConvNext__](https://arxiv.org/pdf/2201.03545.pdf) as a new approach. And luckly, the model reach a strong baseline. The mAP
on DOTA v1.0 dataset achieves about 65%.

Now, I am trying to replace the depthwise convolution in bottleneck by a _MHSA_ block.

---

## Version  
### Ver 1.0, update on 04/22/2022  
1. Simplify and modifiy some codes in "./nets" and "./utils".  
2. Rewrite `configure.py` to globally control the whole project.
3. Codes has been tested on `train.py`. (Training process is OK)  

### Ver 1.1, update on 04/27/2022
1. Add ConvNeXt as backbone structure.  
2. Add backbone.py to manage backbone structures. Make "./nets/yolo.py" more readable.  
3. Training part had been tested.

### Ver 2.0, update on 06/10/2022
1. Re-clustering anchor shapes based on cropped DOTA v1.0.  
2. Add "AttentionNeck" structure.  
3. Del some un-use backbone.  

---

## AttentionNeck (TransNeck)
1. The _depthwise_ convolution in bottleneck provides a kind of __attention-like mechanism__.  
2. Why not just use pure `MHSA` structure?  
3. `AttentionNeck`, replace the depthwise convolution in bottleneck by a `MHSA` structure.  

---

## Code Structure
1. "./nets" contains several backbone structures.  
2. If want to add more backbone structure(s), please add under "./nets".  
3. After add backbone, please re-write a _class_ method about new backbone in "./nets/backbones.py". And add your backbone in BACKBONE_LIST in `configure.py`.  
4. Other part is complete enough. Try not to change them.

---

## Train
1. Change training schedule in "configure.py".  
2. Make sure `CLASSES_PATH` and `DATASET_PATH` and `MODEL_PATH` are matched.
3. Weights file will be stored every epoch. It's okey to interrupt and re-start.
4. When re-start, please change `MODEL_PATH` to the weights file when interrupting, as well as `INIT_EPOCH`.  

---

## Evaluate
1. Change the `MODEL_PATH` to the weight which needs to be evaluate.  
2. Make sure you have already got "test.txt" under dir "./your_dataset_name/ImageSets".  
3. Run `get_map.py`. And pay attention to papram `MINOVERLAP`.  

---

## Dataset
1. Mainly use DOTA v1.0 (perhaps will use v1.5 in the future).  
2. Use the official code to crop DOTA into 960 * 960 sub-images.  

---

## TODO

