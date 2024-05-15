import os

def delete_word_bboxes(filename: str, words_label: str = '1'):
    reader = open(filename, 'r')
    line_bboxes = []

    print(f"Reading {filename}...")

    for line in reader:
        values = line.split(" ")
        if values[0] != words_label:
            line_bboxes.append(line)
    
    print(f"Found {len(line_bboxes)} line bbox values")
    
    print(f"Writing")

    reader.close()
    writer = open(filename, 'w')
    for line in line_bboxes:
        writer.write(line)
    writer.close()

def delete_words_bboxes_from_dir(dir_name: str, words_label: str = '1'):
    for filename in os.listdir(dir_name):
        if not filename.endswith('.txt'):
            continue
        delete_word_bboxes(
            os.path.join(dir_name, filename), 
            words_label
            )
    
def proccess_line_file(filename: str):
    reader = open(filename, 'r')
    lines = []
    for line in reader:
        values = line.split(" ")
        if values[0] == '1':
            values[0] = '0'
        for i in range(1, len(values)):
            if float(values[i]) > 1:
                values[i] = '1.0'
            elif float(values[i]) < 0:
                values[i] = '0.0'
        lines.append(
            " ".join(values)
        )
    reader.close()

    writer = open(filename, 'w')
    for line in lines:
        writer.write(line)
    writer.close()

def process_line_files(dir_name: str = "../datasets/lines/train"):
    for filename in os.listdir(dir_name):
        if not filename.endswith(".txt"):
            continue
        proccess_line_file(
            os.path.join(dir_name, filename)
        )


if __name__ == "__main__":
    # process_line_files("../datasets/lines/validation")
    delete_words_bboxes_from_dir("../line_labels", words_label='1')
