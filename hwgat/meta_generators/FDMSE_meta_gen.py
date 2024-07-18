from math import ceil
import os
import csv
from meta_generator import generate_meta

if __name__ == "__main__":
    root = "/data/datasets"
    data_path = "/data/datasets/FDMSE/FDMSE"
    split_file = "/data/datasets/FDMSE/metadata.csv"

    # initializing the titles and rows list
    header = []
    rows = []

    vocab = []

    id = 0
    
    # reading csv file
    with open(split_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        header = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            vid_path = os.path.join(row[1], row[2])
            try:
                f = open(os.path.join(root, vid_path), 'r')
                f.close()

                cls = row[3]
                if cls not in vocab:
                    vocab.append(cls)
                if len(row[4]) > 1:
                    rows.append(["{:07d}".format(id), vid_path, row[2], cls, row[4]])
                id += 1
            except:
                continue
    # print(rows)
    vocab.sort()

    generate_meta(data_path, rows, vocab)

