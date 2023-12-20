import random
import os
import csv


# 从CSV文件读取数据并处理
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过第一行（标题行）
        for row in csv_reader:
            data.append(row[:2])  # 仅保留前两列
    return data


# 生成阴性数据集
def generate_negative_data(positive_data, possible_data, test_negative_data, pathogen_proteins, host_proteins, count):
    negative_data = []
    pathogen_proteins = list(pathogen_proteins)  # 转换为列表
    host_proteins = list(host_proteins)  # 转换为列表
    while len(negative_data) < count:

        random_pathogen_protein = random.choice(pathogen_proteins)
        random_host_protein = random.choice(host_proteins)
        x = (random_pathogen_protein, random_host_protein)

        # 检查是否在阳性、可能的相互作用、测试阴性数据中
        if x not in positive_data and x not in possible_data and x not in test_negative_data and x not in negative_data:
            negative_data.append(x)
    return negative_data


# 创建目录
def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

vir_res_train = 'dataset/train/virus/all/'
vir_res_test = 'dataset/test/virus/all/'
create_directory(vir_res_train)
create_directory(vir_res_test)

all_df = read_csv('data/positive/virus/all/All.csv')
host_proteins = set(row[1] for row in all_df)

interactions_data = read_csv('data/positive/virus/all/All.csv')
possible_interactions_data = read_csv('data/possible/VirPossibleTable.csv')

# 从CSV文件读取Species测试数据，仅保留前两列
Species_testPositivedata = read_csv('dataset/test/virus/species/TestPositiveAll.csv')
Species_testNegativedata = read_csv('dataset/test/virus/species/TestNegativeAll.csv')
num = int(len(interactions_data) * 0.9)
trainpos_data = random.sample([row for row in interactions_data if row not in Species_testPositivedata], num)

pathogen_proteins = set(row[0] for row in interactions_data)

# 选择测试阳性数据的补集
testpos_data = [row for row in interactions_data if row not in trainpos_data]

# 生成训练集中的阴性数据
trainneg_data = generate_negative_data(trainpos_data, possible_interactions_data, Species_testNegativedata,
                                       pathogen_proteins, host_proteins, len(trainpos_data))

# 生成测试集中的阴性数据
testneg_data = generate_negative_data(testpos_data, possible_interactions_data, trainneg_data,
                                      pathogen_proteins, host_proteins, 10 * len(testpos_data))

# 合并数据集并保存为CSV文件
train_file_path = os.path.join(vir_res_train,'All_train.csv')
test_file_path = os.path.join(vir_res_test, 'All_test.csv')

with open(train_file_path, 'w') as train_file:
    train_file.write('Pathogen Protein ID,Host Protein ID,Interactions\n')
    for row in trainpos_data:
        train_file.write(','.join(row) + ',1\n')  # 1表示阳性样本
    for row in trainneg_data:
        train_file.write(','.join(row) + ',0\n')

with open(test_file_path, 'w') as test_file:
    test_file.write('Pathogen Protein ID,Host Protein ID,Interactions\n')
    for row in testpos_data:
        test_file.write(','.join(row) + ',1\n')  # 1表示阳性样本
    for row in testneg_data:
        test_file.write(','.join(row) + ',0\n')