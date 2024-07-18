import csv
import os
import shutil
import argparse
from meta_generator import generate_meta

def generate_pose_meta(root, dataset, subset, flip=False):
    meta_path = os.path.join(root, dataset + "_meta")
    data_path = os.path.join(root, dataset + "_pose")

    # initializing the titles and rows list
    header = []
    rows = []

    if subset != None:
        meta_file = meta_path + '/metadata' + str(subset) + '.csv'
        class_file = data_path + '_meta/classes' + str(subset) + '.txt'
    else:
        meta_file = meta_path + '/metadata.csv'
        class_file = data_path + '_meta/classes.txt'
    
    # reading csv file
    with open(meta_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        header = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            row[0] = row[0][:-4] + '_pose.avi'
            temp = row[0].split('/')
            temp[0] = temp[0] + '_pose'
            row[0] = os.path.join(*temp)
            try:
                f = open(row[0], 'r')
                f.close()
            except:
                continue
            rows.append(row)
            if flip:
                row_flip = row.copy()
                row_flip[0] = row_flip[0][:-4] + '_flipped.avi'
                try:
                    f = open(row_flip[0], 'r')
                    f.close()
                except:
                    continue
                rows.append(row_flip)
    generate_meta(data_path, rows, subset)


    shutil.copyfile(meta_path + '/classes.txt', class_file)