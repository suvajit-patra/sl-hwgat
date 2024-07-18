from math import ceil
import os
import csv
from meta_generator import generate_meta

if __name__ == "__main__":
    data_path = "datasets/AUTSL"
    folder_name = "AUTSL/"
    class_path = "datasets/assets/autsl_metadata/AUTSL/SignList_ClassId_TR_EN.csv"
    split_path = "datasets/assets/autsl_metadata/AUTSL/"

    # initializing the titles and rows list
    header = []
    rows = []

    vocab = []

    # reading csv file
    with open(class_path, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        header = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            vocab.append(row[2].strip().lower())

    id = 0
    for split in ['train', 'test', 'val']:
        # reading csv file
        with open(split_path + '/'+split+'_labels.csv', 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            
            # extracting field names through first row
            header = next(csvreader)
        
            # extracting each data row one by one
            for row in csvreader:
                vid_path = os.path.join(folder_name, split, row[0] + '_color.mp4')
                rows.append(["{:07d}".format(id), vid_path, row[0] + '_color.mp4', vocab[int(row[1])], split])

                id += 1


    vocab.sort()

    generate_meta(data_path, rows, vocab)

