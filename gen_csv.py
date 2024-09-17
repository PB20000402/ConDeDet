from argparse import ArgumentParser
from src.data_generator import build_PDG, build_XFG
import networkx as nx
from src.preprocess.symbolizer import clean_gadget, tokenize_code_line
from src.utils import filter_warnings
from os import system
import matplotlib.pyplot as plt
from typing import List, Dict
from os.path import join, exists
import os


def pdg_to_text(pdg_graph):
    text_representation = []
    for node in pdg_graph:
        neighbors = pdg_graph.neighbors(node)
        for neighbor in neighbors:
            edge_type = pdg_graph[node][neighbor]['c/d']
            text_representation.append(f"({node}, {neighbor}): {edge_type}")
    return text_representation


def main():
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-j",
                              "--joern_path",
                              help="joern_path",
                              default="/home/thinkstation02/Desktop/hxfile/joern/joern-parse",
                              type=str)
    __arg_parser.add_argument("-f",
                              "--file_name",
                              help="file_name",
                              default="train-dataset",
                              type=str)
    __args = __arg_parser.parse_args()
    filter_warnings()

    joern = __args.joern_path
    root = join("data", "demo")
    source_path = join(root, "train-dataset")
    print("source_path:", source_path)
    csv_out_path = join(root, "csv", __args.file_name)
    print("csv_out_path:", csv_out_path)
    print("__args.file_name:", __args.file_name)
    system(f"{joern} {csv_out_path} {source_path}")

    out_root_path = join(root, "PDG-txt")

    csv_path = join(root, "csv", __args.file_name, )
    print("csv_path:", csv_path)

    source_file_path = join(source_path, __args.file_name)

    pdg_out_path = join(out_root_path, __args.file_name, "pdg.txt")

    # preprocess for the code to detect
    PDG, key_line_map = build_PDG(csv_path, f"data/sensiAPI.txt", source_file_path)

    if not exists(join(out_root_path, __args.file_name)):
        os.makedirs(join(out_root_path, __args.file_name))

    # Convert PDG graph to text representation
    pdg_text = pdg_to_text(PDG)

    # Save PDG text to a text file
    with open(pdg_out_path, 'w', encoding='utf-8') as f:
        for line in pdg_text:
            f.write(line + '\n')

    print(f"成功保存到 {pdg_out_path}")


if __name__ == '__main__':
    main()
