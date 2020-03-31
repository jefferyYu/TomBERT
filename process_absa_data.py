import os
import sys
import argparse

__author__ = "Jianfei"

TASKS = ["twitter", "twitter2015"]

def format_absa(data_dir, task):
    print("Processing..."+task)
    absa_dir = os.path.join(data_dir, task)
    if not os.path.isdir(absa_dir):
        os.mkdir(absa_dir)

    absa_train_file = os.path.join(absa_dir, "train.txt")
    absa_test_file = os.path.join(absa_dir, "test.txt")

    assert os.path.isfile(absa_train_file), "Train data not found at %s" % absa_train_file
    assert os.path.isfile(absa_test_file), "Test data not found at %s" % absa_test_file

    fin = open(absa_train_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(absa_dir, "train.tsv"), 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 String\t#2 String\n")
        count = 0
        for i in range(0, len(lines), 4):
            count += 1
            text = lines[i].strip()
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()
            label = str(int(polarity) + 1)
            train_fh.write("%s\t%s\t%s\t%s\n" % (count, label, text, aspect))

    fin = open(absa_test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(absa_dir, "dev.tsv"), 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 String\t#2 String\n")
        count = 0
        for i in range(0, len(lines), 4):
            count += 1
            text = lines[i].strip()
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()
            label = str(int(polarity) + 1)
            train_fh.write("%s\t%s\t%s\t%s\n" % (count, label, text, aspect))

    fin = open(absa_test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(absa_dir, "test.tsv"), 'w') as test_fh:
        test_fh.write("index\t#1 Label\t#2 String\t#2 String\n")
        count = 0
        for i in range(0, len(lines), 4):
            count += 1
            text = lines[i].strip()
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()
            label = str(int(polarity) + 1)
            test_fh.write("%d\t%s\t%s\t%s\n" % (count, label, text, aspect))
    print("\tCompleted!")

def format_abmsa(data_dir, task):
    print("Processing..."+task)
    absa_dir = os.path.join(data_dir, task)
    if not os.path.isdir(absa_dir):
        os.mkdir(absa_dir)

    absa_train_file = os.path.join(absa_dir, "train.txt")
    absa_dev_file = os.path.join(absa_dir, "dev.txt")
    absa_test_file = os.path.join(absa_dir, "test.txt")

    assert os.path.isfile(absa_train_file), "Train data not found at %s" % absa_train_file
    assert os.path.isfile(absa_test_file), "Test data not found at %s" % absa_test_file

    fin = open(absa_train_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(absa_dir, "train.tsv"), 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 ImageID\t#3 String\t#3 String\n")
        count = 0
        for i in range(0, len(lines), 4):
            count += 1
            text = lines[i].strip()
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()
            imgid = lines[i + 3].strip()
            label = str(int(polarity) + 1)
            train_fh.write("%d\t%s\t%s\t%s\t%s\n" % (count, label, imgid, text, aspect))

    fin = open(absa_dev_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(absa_dir, "dev.tsv"), 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 ImageID\t#3 String\t#3 String\n")
        count = 0
        for i in range(0, len(lines), 4):
            count += 1
            text = lines[i].strip()
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()
            imgid = lines[i + 3].strip()
            label = str(int(polarity) + 1)
            train_fh.write("%d\t%s\t%s\t%s\t%s\n" % (count, label, imgid, text, aspect))

    fin = open(absa_test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(os.path.join(absa_dir, "test.tsv"), 'w') as test_fh:
        test_fh.write("index\t#1 Label\t#2 ImageID\t#2 String\t#2 String\n")
        count = 0
        for i in range(0, len(lines), 4):
            count += 1
            text = lines[i].strip()
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()
            imgid = lines[i + 3].strip()
            label = str(int(polarity) + 1)
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (count, label, imgid, text, aspect))
    print("\tCompleted!")


def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='absa_data')
    parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='all')
    parser.add_argument('--path_to_mrpc', help='path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt',
                        type=str, default='')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == 'restaurant' or task == 'laptop' or task == 'twitter2014':
            format_absa(args.data_dir, task)
        else:
            format_abmsa(args.data_dir, task)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))