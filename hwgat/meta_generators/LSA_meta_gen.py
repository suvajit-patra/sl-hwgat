from math import ceil
import os
import csv
from meta_generator import generate_meta


train = ['001', '002', '003', '004', '005', '006', '007']
val = ['008']
test = ['009', '010']

if __name__ == "__main__":
    root = '/data2/datasets/LSA64'
    data_path = os.path.join(root, "LSA64")
    class_path = os.path.join(root, "lsa64_signs.md")
    meta_path = os.path.join(root, 'LSA64_meta/')

# words = {}
# with open('../word_tags.csv', 'r') as file:
#     csvFile = csv.reader(file)

#     for line in csvFile:
#         words[line[1]] = line[0]
    rows = []

    vocab = []

    id = 0
    with open(class_path, 'r') as file:
        # creating a csv reader object
        lines = file.readlines()
    
        # extracting each data row one by one
        for row in lines:
            vocab.append(row.split("|")[1].strip().lower())

    for video in os.listdir(data_path):
        
        sub_name = video.split('_')[1]
        split = None
        if sub_name in train:
            split = 'train'
        if sub_name in test:
            split = 'test'
        if sub_name in val:
            split = 'val'
        
        rows.append(["{:07d}".format(id), os.path.join(data_path, video), video, vocab[int(video.split('_')[0])-1], split])
        id += 1

    vocab.sort()
    generate_meta(data_path, rows, vocab)