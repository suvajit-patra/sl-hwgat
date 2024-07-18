import os
import csv

def generate_meta(data_path, rows, vocab, subset=None):
    try:
        os.makedirs(data_path + "_meta")
    except:
        pass

    if subset != None:
        meta_file = data_path + '_meta/metadata_' + str(subset) + '.csv'
        class_file = data_path + '_meta/classes_' + str(subset) + '.txt'
    else:
        meta_file = data_path + '_meta/metadata.csv'
        class_file = data_path + '_meta/classes.txt'

    header = ['id', 'video_dir', 'video_name', 'class', 'split']
    
    with open(meta_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)

    if vocab:
        with open(class_file, 'w') as f:
            for word in vocab:
                f.write(word + "\n")
    