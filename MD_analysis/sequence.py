# 读取 PDB 文件
import pandas as pd

with open('./com.pdb', 'r') as f:
    amber_lines = f.readlines()

with open('./protein.pdb', 'r') as f:
    pre_lines = f.readlines()

# 定义一个函数来提取氨基酸编号和对应的信息
def extract_aa_info(lines):
    aa_info = {}
    for index,line in enumerate(lines):
        # PDB 格式中，氨基酸信息通常在第 23 到 26 列（Amino acid residue number）出现
        if line.startswith("ATOM") or line.startswith("HETATM"):
            res_num = int(line[22:26].strip())  # 提取氨基酸编号
            res_name = line[17:20].strip()      # 提取氨基酸名称（如 ALA, GLY 等）
            aa_info[res_num] = res_name
    return aa_info

def extract_preaa_info(lines):
    aa_info = {}
    for index, line in enumerate(lines):
        # PDB 格式中，氨基酸信息通常在第 23 到 26 列（Amino acid residue number）出现
        if line.startswith("ATOM") or line.startswith("HETATM"):
            res_num = int(line[22:26].strip())  # 提取氨基酸编号
            res_name = line[17:20].strip()  # 提取氨基酸名称（如 ALA, GLY 等）
            chain_id = line[21].strip()  # 提取链 ID
            if chain_id == 'A':
                aa_info[res_num] = res_name
    return aa_info


# 提取两个文件中的氨基酸信息
amber_aa_info = extract_aa_info(amber_lines)
pre_aa_info = extract_preaa_info(pre_lines)

# 打印提取的氨基酸信息
print("Amber file amino acid info:")
print(amber_aa_info)

print("Pre file amino acid info:")
print(pre_aa_info)

# amber_index = list(amber_aa_info.keys())[:len(amber_aa_info)-1]
# pre_index = list(pre_aa_info.keys())
# amide = list(amber_aa_info.values())[:len(amber_aa_info)-1]

amber_index = list(amber_aa_info.keys())[:306]
pre_index = list(pre_aa_info.keys())
amide = list(amber_aa_info.values())[:306]

df = pd.DataFrame({
    'Amber Index': amber_index,
    'Pre Index': pre_index,
    'Amino Acid': amide
})
df.to_csv('./cdk9_sequence.csv', index=False)
# 打印 DataFrame
print(df)
