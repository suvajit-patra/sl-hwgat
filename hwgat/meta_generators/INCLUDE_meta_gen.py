# Copy Train_Test_Split INCLUDE folder to ./ to run this code


from math import ceil
import random
import os
import csv
from meta_generator import generate_meta

val_split = 0.1

if __name__ == "__main__":
    root = '/data2/datasets/INCLUDE'
    data_path = os.path.join(root, "INCLUDE")
    split_path = os.path.join(root, "Train_Test_Split")

    # initializing the titles and rows list
    header = []
    rows = []
    vocab = []
    
    train_rows = {}
    # reading csv file
    with open(split_path + '/train_include.csv', 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        header = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            vid_path = os.path.join(data_path, row[3])
            try:
                f = open(vid_path, 'r')
                f.close()

                if 'Extra' in vid_path or 'extra' in vid_path:
                    cls = vid_path.split('/')[-3].split('.')[1].strip().lower()
                else:
                    cls = vid_path.split('/')[-2].split('.')[1].strip().lower()
                if cls not in vocab:
                    vocab.append(cls)

                if train_rows.get(cls) == None:
                    train_rows[cls] = []
                train_rows[cls].append([os.path.join('INCLUDE', row[3]), vid_path.split('/')[-1], cls, 'train'])
            except Exception as e:
                print(vid_path, e)
                continue
            
    vocab.sort()

    for cls in train_rows.keys():
        idxs = random.sample(range(len(train_rows[cls])), ceil(len(train_rows[cls])*val_split))
        for idx in idxs:
            train_rows[cls][idx][3] = 'val'
    
    id = 0
    for cls in train_rows.keys():
        for row in train_rows[cls]:
            rows.append(["{:07d}".format(id), row[0], row[1], row[2], row[3]])
            id += 1

    test_rows = []

    # reading csv file
    with open(split_path + '/test_include.csv', 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        header = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            vid_path = os.path.join(data_path, row[3])
            try:
                f = open(vid_path, 'r')
                f.close()

                if 'Extra' in vid_path or 'extra' in vid_path:
                    cls = vid_path.split('/')[-3].split('.')[1].strip().lower()
                else:
                    cls = vid_path.split('/')[-2].split('.')[1].strip().lower()
                test_rows.append([vid_path, cls])
                rows.append(["{:07d}".format(id), os.path.join('INCLUDE', row[3]), vid_path.split('/')[-1], cls, 'test'])
                id += 1
            except:
                continue

    generate_meta(data_path, rows, vocab)

