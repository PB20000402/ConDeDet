import os
from collections import defaultdict

# Function to add comments to code lines based on dependencies
def add_comments_to_code(code_path, pdg_path, output_dir):
    try:
        with open(code_path, 'r', encoding='utf-8') as f:
            code_lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(code_path, 'r', encoding='GBK') as f:  # 使用GBK编码尝试解码
                code_lines = f.readlines()
        except UnicodeDecodeError as e:
            print(f"Error decoding {code_path}: {e}")
            return  # 处理解码错误，可能放弃处理该文件或者采取其他措施

    # Read the PDG file
    with open(pdg_path, 'r', encoding='utf-8') as f:
        pdg_lines = f.readlines()

    # Parse the dependencies
    dependencies = defaultdict(list)
    line_numbers = set()

    for line in pdg_lines:
        line = line.strip()
        if line:  # Ignore empty lines
            dep_tuple, dep_type = line.split(": ")
            line_from, line_to = map(int, dep_tuple.strip("()").split(", "))
            line_numbers.add(line_from)
            line_numbers.add(line_to)

            if dep_type == "c":
                dep_type_str = "data depends on"
            elif dep_type == "d":
                dep_type_str = "control depends on"
            else:
                raise ValueError(f"Unknown dependency type {dep_type}")

            dependencies[line_to].append(f"line {line_to} {dep_type_str} line {line_from}")

    # Modify the code lines by adding the comments
    for line_no, deps in dependencies.items():
        comment = f" //this line is line {line_no};" + "; ".join(deps)
        code_lines[line_no - 1] = code_lines[line_no - 1].rstrip() + comment + "\n"  # -1 because lines are 0-indexed

    # Add "//第*行代码：" to the end of lines that appear in the PDG.txt but have no dependencies
    for line_no in line_numbers:
        if line_no not in dependencies:
            code_lines[line_no - 1] = code_lines[line_no - 1].rstrip() + f" //this line is line {line_no};" + "\n"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the filename from the input code path
    code_filename = os.path.basename(code_path)

    # Change the file extension for the output file to .c
    output_file_path = os.path.join(output_dir, code_filename)

    # Save the modified code to a new file with .c extension
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in code_lines:
            f.write(line)

# Process all .c files and their corresponding PDG.txt files in the source directory
source_dir = "data/demo/train-dataset"
output_dir = "data/demo/train-comment-dataset"

for root, _, files in os.walk(source_dir):
    for filename in files:
        if filename.endswith(".cpp"):
            code_path = os.path.join(root, filename)
            pdg_prefix = ".pdg"
            pdg_filename = filename + pdg_prefix + ".txt"
            pdg_path = os.path.join(root, pdg_filename)
            add_comments_to_code(code_path, pdg_path, output_dir)
