# to run this code you need to copy ./WLASL/code/I3D/preprocess to ./



import json
import os
import argparse
import csv
import numpy as np
from meta_generator import generate_meta

def arg_parser():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--s', type=int, default=1000, help='Enter the subset(100, 200, 500, 1000) here')
    # Parse the argument
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parser()
    subset = args.s
    root = "/data3/datasets/"
    json_path = 'datasets/assets/msasl_metadata/MS_ASL_SI_ASL' + str(subset) + '_'
    classes_path = 'datasets/assets/msasl_metadata/MSASL_classes.json'
    data_path = 'datasets/MSASL'

    rows = []
    vocab = {}

    # Opening txt file
    with open(classes_path) as file:
        vocab = json.load(file)

    vocab.sort()

    id = 0

    if subset != 1000:
        full_dataset_rows = {}
        try:
            with open(os.path.join(data_path, 'MSASL_meta', 'metadata.csv')) as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                for row in csv_reader:
                    full_dataset_rows[row[0]] = row

        except:
            print('generate for 1000 first')
            exit()


    # Opening JSON file
    arr = np.zeros(len(vocab))
    for split in ['train', 'test', 'val']:
        with open(json_path+split+'.json') as file:
            # returns JSON object as
            # a dictionary
            data = json.load(file)

            # Iterating through the json
            # list
            if subset != 1000:
                for entry in data:
                    # print(vocab[data[entry]['action'][0]])
                    try:
                        rows.append(full_dataset_rows[os.path.join('MSASL', split, entry['clean_text'], str(entry['signer_id'])+'.mp4')])
                    except:
                        pass
            else:     
                for entry in data:
                    if entry['text'] in vocab:
                        rows.append(["{:07d}".format(id), os.path.join('MSASL', split, entry['clean_text'], str(entry['signer_id'])+'.mp4'), str(entry['signer_id'])+'.mp4',
                                        entry['text'], split])
                        arr[vocab.index(entry["text"])] += 1
                        id += 1
                    else:
                        print(entry['text'])
                        print("gg", vocab[int(np.where(arr==0)[0][0])])
    

    # print(len(rows))
    # exit()
    # vocabs_list = list(vocab.values())
    # vocabs_list.sort()


    if subset == 1000:
        generate_meta(data_path, rows, vocab)
    else:
        generate_meta(data_path, rows, vocab, subset=subset)

