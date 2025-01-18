import os
import csv
import random

test_file_path = "/home/coder/project/data/_downstream_data/rs/amazonqa/test_all.txt"

with open(test_file_path, "r") as test_file:
    test_reader = csv.reader(test_file, delimiter="\t")
    all_data = list(test_reader)

random.seed(42)

sample_size = min(300000, len(all_data))
selected_indices = random.sample(range(len(all_data)), sample_size)
selected_data = [all_data[i] for i in selected_indices]

output_file_path = os.path.join("/home/coder/project/data/_downstream_data/rs/amazonqa/test.txt")

with open(output_file_path, "w", newline="") as output_file:
    test_writer = csv.writer(output_file, delimiter="\t")
    test_writer.writerows(selected_data)

print(f"selected data is saved: {output_file_path}")
