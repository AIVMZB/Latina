import os
import shutil
import re
import csv


def create_dataset_structure(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            class_name = re.sub(r'^\d+-', '', os.path.splitext(file)[0].lower())
            class_dir = os.path.join(target_dir, class_name)

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            src_path = os.path.join(root, file)
            dst_path = os.path.join(class_dir, file)
            shutil.copy(src_path, dst_path)


def count_files_in_classes(target_dir, csv_file):
    class_counts = {}

    for class_name in os.listdir(target_dir):
        class_path = os.path.join(target_dir, class_name)
        if os.path.isdir(class_path):
            file_count = len(os.listdir(class_path))
            class_counts[class_name] = file_count

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'Files'])
        for class_name, count in class_counts.items():
            writer.writerow([class_name, count])


source_folder = 'word_dataset'
target_folder = 'sorted_dataset'
csv_output_file = 'class_distribution.csv'

create_dataset_structure(source_folder, target_folder)

count_files_in_classes(target_folder, csv_output_file)

print(f"Class distribution has been saved to {csv_output_file}")

