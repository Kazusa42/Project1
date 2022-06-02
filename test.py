import os

c_annotation_files_dir = r'C:\Users\lyin0\Desktop\DOTA\Color_Annotations_XML\\'

for file_name in os.listdir(c_annotation_files_dir):
    new_name = r'c_' + file_name
    os.rename(c_annotation_files_dir + file_name, c_annotation_files_dir + new_name)
