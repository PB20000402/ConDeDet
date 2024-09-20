from argparse import ArgumentParser
from src.data_generator import build_PDG, build_XFG
import networkx as nx
from src.preprocess.symbolizer import clean_gadget, tokenize_code_line
from src.utils import filter_warnings
from os import system
import matplotlib.pyplot as plt
from typing import List, Dict
from os.path import join, exists, dirname
import os


def pdg_to_text(pdg_graph):
    """
    将PDG图转换为文本表示.

    参数:
        pdg_graph (networkx.DiGraph): PDG图.

    返回:
        List[str]: PDG的文本表示列表.
    """
    text_representation = []
    for node in pdg_graph:
        neighbors = pdg_graph.neighbors(node)
        for neighbor in neighbors:
            edge_type = pdg_graph[node][neighbor]['c/d']
            text_representation.append(f"({node}, {neighbor}): {edge_type}")
    return text_representation


def generate_pdg_text(csv_path_root: str, source_file_root: str) -> List[List[str]]:
    """
    自动生成多个PDG组合并生成文本表示.

    参数:
        csv_path_root (str): csv_path的根目录.
        source_file_root (str): source_file_path的根目录.

    返回:
        List[List[str]]: 多个PDG的文本表示列表的列表.
    """
    pdg_texts_list = []
    for csv_subfolder in os.listdir(csv_path_root):
        csv_subfolder_path = os.path.join(csv_path_root, csv_subfolder)
        if not os.path.isdir(csv_subfolder_path):
            continue

        # 在csv子目录下查找csv文件
        for csv_filename in os.listdir(csv_subfolder_path):
            csv_path = os.path.join(csv_subfolder_path, csv_filename)

            # 查找对应的源代码文件
            source_subfolder = os.path.join(source_file_root, csv_subfolder)
            for source_filename in os.listdir(source_subfolder):
                source_file_path = os.path.join(source_subfolder, source_filename)
                source_code_folder = os.path.dirname(source_file_path)

                # 调试信息
                print("处理 csv 文件:", csv_path)
                print("处理源代码文件:", source_file_path)

                PDG, _ = build_PDG(csv_path, f"data/sensiAPI.txt", source_file_path)
                if PDG is None:
                    print("生成 PDG 图失败")
                    continue

                pdg_text = pdg_to_text(PDG)

                # 生成PDG文本文件，保存在对应的source_file目录下
                output_file = os.path.join(source_code_folder, f"{source_filename}.pdg.txt")
                with open(output_file, 'w') as f:
                    f.write("\n".join(pdg_text))

                print("PDG文本已保存到文件:", output_file)
                pdg_texts_list.append(pdg_text)

    return pdg_texts_list


if __name__ == '__main__':
    # 自动生成组合
    csv_path_root = "data/demo/csv/train-dataset/home/thinkstation02/Desktop/hxfile/DeepWukong/data/demo/train-dataset"
    source_file_root = "data/demo/train-dataset"

    pdg_texts_list = generate_pdg_text(csv_path_root, source_file_root)
