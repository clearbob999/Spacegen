import argparse
import multiprocessing as mp

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from utils import *
from break_ligand_5 import CompoundSplit


def process_data(file_path):
    Data_df = pd.read_csv(file_path, encoding='ISO-8859-1')
    # Data_df_ic = Data_df[(Data_df['Standard Value'] <= 35) & (Data_df['Molecular Weight'] <= 550)].drop_duplicates('Molecule ChEMBL ID').dropna(subset=['Target Name'])
    if 'CDK9' in file_path:
        Data_df_ic = Data_df[(Data_df['Standard Value'] <= 50) & (Data_df['Molecular Weight'] <= 600)].drop_duplicates(
            'Molecule ChEMBL ID').dropna(subset=['Target Name'])
        Data_filter = Data_df_ic[Data_df_ic['Target Name'].str.contains('9')]
    elif 'FGFR4' in file_path:
        Data_df_ic = Data_df[(Data_df['Standard Value'] <= 50) & (Data_df['Molecular Weight'] <= 600)].drop_duplicates(
            'Molecule ChEMBL ID').dropna(subset=['Target Name'])
        Data_filter = Data_df_ic[Data_df_ic['Target Name'].str.contains('4')]
    elif 'KIF18A' in file_path:
        Data_df_ic = Data_df[(Data_df['Standard Value'] <= 200) & (Data_df['Molecular Weight'] <= 600)].drop_duplicates(
            'Molecule ChEMBL ID').dropna(subset=['Target Name'])
        Data_filter = Data_df_ic[Data_df_ic['Target Name'].str.contains('18A')]
    else:
        Data_df_ic = Data_df[(Data_df['Standard Value'] <= 50) & (Data_df['Molecular Weight'] <= 600) & (Data_df['Standard Units'] == 'nM')].drop_duplicates(
            'Molecule ChEMBL ID').dropna(subset=['Target Name'])
        Data_filter = Data_df_ic[Data_df_ic['Target Name'].str.contains('Rho-associated protein kinase 1')]
    # return Data_filter['Smiles'].to_list()
    return Data_filter

# SMILES 转化为 RDKit 分子对象的函数
def smiles_to_mol(smiles_list):
    mols = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            mols.append(mol)
    return mols

def worker(smi,args):
    return get_morgen_fingerprint(smi, nBits=args.feature)

def parse_args():
    parser = argparse.ArgumentParser(description='Ligand Fragmentation')
    parser.add_argument('-i', '--input', type=str, default='FC1=CC(OC)=C(C2=C(Cl)C=NC(NC3=NN(C4CCNCC4)C=C3)=C2)C=C1',
                        help='ligand SMILES')
    parser.add_argument('-b', '--BB_data', type=str, choices=['Enamine BB Catalog', 'Enamine stock', 'mcule'],
                        default='Enamine BB Catalog', help='Brick stock data')
    parser.add_argument('-m', '--minFragmentSize', type=int, default=4,
                        help='Minimum fragment size')
    parser.add_argument('-c', '--cut_num', type=int, default=2,
                        help='Ligand fragmentation level (recommended 1 or 2). If molecular weight is greater than 500, it is better to set it to 2.')
    parser.add_argument('-r', '--onlyUseReactions', type=list, default=None,
                        help='Fragmentation rules to be used')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Similarity search threshold (value between 0 and 1)')
    parser.add_argument('-f', '--fingerprint', type=str, choices=['Morgan','Maccs','Pharm2D'],default='Morgan',
                        help='Fingerprint type')
    parser.add_argument('--feature', type=int, default=256)
    args = parser.parse_args()
    return args

def process(rules, match_bbs, filter_rules_path, match_bb_path, rule_json_path):
    process_reac_file(rules, match_bbs, filter_rules_path, match_bb_path=match_bb_path, rule_json_path=rule_json_path)

    with open(match_bb_path, 'r') as f:
        data = [l.strip().split()[0] for l in f.readlines()]
    print('the number of fragements =', len(data))

    with mp.Pool(processes=20) as pool:
        # embeddings = pool.map(worker, data)
        embeddings = pool.starmap(worker, [(smi, args) for smi in data])

    # with open(args.output, 'wb') as f:
    #     pickle.dump(embeddings, f)
    np.save(npy_output_path, np.array(embeddings))

if __name__ == '__main__':
    # CDK9_file = r'./CDK9/CDK9.csv'
    # FGFR4_file = r'./FGFR4/FGFR4.csv'
    target_name = "ROCK1"
    file_path = f'./{target_name}/{target_name}.csv'
    output_path = file_path.replace('.csv', '_valid_molecules.sdf')
    frag_path = f'data/{target_name}_query_frags_rule.txt'
    filter_rules_path = f'data/{target_name}_rxn_filter_rule.txt'
    match_bb_path = f'data/{target_name}_matched_bbs_rule.txt'
    npy_output_path = f'data/{target_name}_matched_bbs_emb_256_rule.npy'
    rule_json_path = f'data/{target_name}_data_for_reaction_filtered_rule.json.gz'

    sdf_writer = Chem.SDWriter(output_path)
    data_filter = process_data(file_path)
    data_filter['Standardized_Smiles'] = data_filter['Smiles'].apply(standardize_smi)
    #按standard value升序排
    data_filter = data_filter.sort_values(by='Standard Value')

    mols = []
    break_result_lst = []

    args = parse_args()
    ts = CompoundSplit(args)

    for i in data_filter.iterrows():
        try:
            # print(i[1]['Standardized_Smiles'])
            mol = Chem.MolFromSmiles(i[1]['Standardized_Smiles'])
            # 写入分子名字和ic50数据信息
            mol.SetProp('_Name', i[1]['Molecule ChEMBL ID'])
            mol.SetProp('_IC50', str(i[1]['Standard Value']))
            # print(ts.decompose(mol))
            break_result_lst.extend(ts.decompose(mol))
            sdf_writer.write(mol)
        except Exception as e:
            print(e)
            print(i)
            continue

    sdf_writer.close()
    # 将分解结果写入文件    去重少了一半多
    break_result_lst = list(set(break_result_lst))

    replace_result, search_result, dfs = ts.replace_dummys(break_result_lst)

    query_frags = []
    match_bbid = []
    match_bbs = set()
    for key, value in replace_result.items():
        query_frags += value
    with open(frag_path, 'w') as f:
        for frag in query_frags:
            # smiles写法标准化
            try:
                mol = Chem.MolFromSmiles(frag)  # 转化为分子对象
                frag = Chem.MolToSmiles(mol)
                f.write(frag + '\n')  # 写入文件
                # if mol:
                #     mol_weight = rdMolDescriptors.CalcExactMolWt(mol)  # 计算分子量
                #     if mol_weight > 300:  # 分子量过滤
                #         continue
                #     frag = Chem.MolToSmiles(mol)  # 转化为标准化 SMILES
                    # match_bbs.add(frag)  # 加入匹配片段集合
                    # f.write(frag + '\n')  # 写入文件
            except Exception as e:
                print(e)
                continue

    for key, value in search_result.items():
        match_bbid += value
    match_bbid = list(set(match_bbid))
    print('query_frags :', len(query_frags))
    print('match_bbid :', len(match_bbid))

    bb_data = pd.read_feather('../../Data/2_rules/enamine_bbcat_to_rules.feather')
    # 根据match_bbs中的ID从bb_data中提取smiles
    for i in match_bbid:
        bb_smiles = bb_data.loc[bb_data['ID'] == i]['SMILES'].values[0]
        mol = Chem.MolFromSmiles(bb_smiles)
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        if mol_weight > 300:
            continue
        match_bbs.add(bb_data.loc[bb_data['ID'] == i]['SMILES'].values[0])

    match_bbs = list(match_bbs)
    print('match_bbs :', len(match_bbs))

    rule_info = pd.read_excel('../../Data/2_rules/rules.xlsx')
    rules = rule_info['Smirks'].tolist()

    with open('../../Data/2_rules/rxn_set_uspto.txt', 'r') as f:
        uspto_rules = [l.strip() for l in f.readlines()]

    process(rules, match_bbs, filter_rules_path, match_bb_path, rule_json_path)
    process(uspto_rules, match_bbs, filter_rules_path, match_bb_path, rule_json_path)
