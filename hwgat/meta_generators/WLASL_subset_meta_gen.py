# to run this code you need to copy ./WLASL/code/I3D/preprocess to ./



import json
import os
import argparse
import csv
from meta_generator import generate_meta

def arg_parser():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--s', type=int, default=2000, help='Enter the subset(100, 300, 1000, 2000) here')
    # Parse the argument
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parser()
    subset = args.s
    root = "/data2/datasets/WLASL"
    json_path = root+'/preprocess/nslt_' + str(subset) + '.json'
    classes_path = root+'/preprocess/wlasl_class_list.txt'
    data_path = root+"/WLASL"

    rows = []
    vocab = []

    # Opening txt file
    with open(classes_path) as file:
        for line in file:
            vocab.append(line.split('\t')[1].strip())

    vocab = list((map(lambda x: x.lower(), vocab[0:subset])))
    vocab.sort()

    id = 0

    if subset != 2000:
        full_dataset_rows = {}
        try:
            with open(os.path.join(root, "WLASL_meta", 'metadata.csv')) as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                for row in csv_reader:
                    full_dataset_rows[row[2]] = row

        except:
            print('generate for 2000 first')
            exit()


    # Opening JSON file
    with open(json_path) as file:
        # returns JSON object as
        # a dictionary
        data = json.load(file)

        # Iterating through the json
        # list
        if subset != 2000:
            for entry in data.keys():
                # print(vocab[data[entry]['action'][0]])
                rows.append(full_dataset_rows[entry +'.mp4'])
        else:     
            for entry in data.keys():
                rows.append(["{:07d}".format(id), os.path.join('WLASL', entry +'.mp4'), entry +'.mp4',
                                    vocab[data[entry]['action'][0]], 
                                    data[entry]['subset'] if data[entry]['subset'] != 'val' else 'val'])
                id += 1

    # print(len(rows))
    # exit()

    if subset == 2000:
        generate_meta(data_path, rows, vocab)
    else:
        generate_meta(data_path, rows, vocab, subset=subset)
