import os
import shutil
from distutils.dir_util import copy_tree

Main_data_dir  = "../CarND-Behavioral-Cloning-P3_data/"
folders = ["data0", "data1", "data2"]
log_file_name = "driving_log.csv"
output_data_path = Main_data_dir + "data/"
output_images_path = output_data_path + "IMG/"
output_log_file_name = output_data_path + log_file_name
First = True

merge_csv_files = True
merge_img_folders = True

csv_files = []
images_folders = []
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)

if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)
for data_folder in folders:
    loc_output_data_path = Main_data_dir + data_folder +"/"
    loc_output_images_path = loc_output_data_path + "IMG/"
    loc_csv_file = loc_output_data_path + log_file_name
    csv_files.append(loc_csv_file)
    images_folders.append(loc_output_images_path)

if merge_csv_files:
        removed_str = b',,,,,,\r\n'
        with open(output_log_file_name, "wb") as outfile:
            for f in csv_files:
                with open(f, "rb") as infile:
                    if First:
                        lines = infile.readlines()[:]
                        lines = list(filter(lambda a: a != removed_str, lines))
                        outfile.writelines(lines)
                        First = False
                    else:
                        lines = infile.readlines()[1:]
                        lines = list(filter(lambda a: a != removed_str, lines))
                        outfile.writelines(lines)


if merge_img_folders:
    for images_folder in images_folders:
        print("copying folder: " + images_folder)
        #copy_tree(images_folder, output_images_path, preserve_mode=False, verbose=1)
        cmd = "xcopy /D /Y %s %s" %(os.path.abspath(images_folder), os.path.abspath(output_images_path))
        cmd.replace("/", "\\")
        #print(cmd)
        os.system(cmd)
