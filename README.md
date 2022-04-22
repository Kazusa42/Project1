# Project1
Code for thesis.  
Mainly use YOLO v4 to do detection.

## Version  
### Ver 1.0, update on 04/22/2022  
1. Simplify and modifiy some codes in "./nets" and "./utils".  
2. Rewrite "configure.py" to globally control the whole project.
3. Codes has been tested on train.py. (Training process is OK)  

## Code Structure
1. "./nets" contains several backbone structures.  
2. If want to add more backbone structure(s), please add under "./nets".  
3. Other part is complete enough. Try not to change them.

## Train
1. Change training schedule in "configure.py".  
2. Make sure CLASSES_PATH and DATASET_PATH and MODEL_PATH are matched.
3. Weights file will be stored every epoch. It's okey to interrupt and re-start.
4. When re-start, please change MODEL_PATH to the weights file when interrupting, as well as INIT_EPOCH.  

## Evaluate
1. Change the MODEL_PATH to the weight which needs to be evaluate.  
2. Make sure you have already got "test.txt" under dir "./your_dataset_name/ImageSets".  
3. Run get_map.py. And pay attention to papram _MINOVERLAP_.  
