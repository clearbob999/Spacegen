import argparse
import ast
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pandarallel import pandarallel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Recap, AllChem, DataStructs, Descriptors, MACCSkeys, QED
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from tqdm import tqdm
from itertools import chain
from scripts import Bricap
import time
import re
import datetime
from rdkit.Contrib.SA_Score import sascorer
import seaborn as sns

environs = {
    '[1*]' : ['[OH]'],
    '[2*]' : [''],
    '[3*]' : ['[OH]'],
    '[4*]' : ['[OH]'],
    '[5*]' : ['[H]'],
    '[6*]' : ['[Cl]'],
    '[7*]' : ['[Cl]'],
    '[8*]' : [''],
    '[9*]' : ['[O]=[C]='],
    '[10*]' : [''],
    '[11*]' : ['[OH]'],
    '[12*]' : ['[Cl]'],
    '[13*]' : ['[O]='],
    '[14*]' : ['[Cl]'],
    '[15*]' : [''],
    '[16*]' : ['[Cl]'],
    '[17*]' : [''],
    '[18*]' : ['[Cl]'],
    '[19*]' : ['OB(O)'],
    '[20*]' : ['[Cl]','[Br]','[I]'],
    '[21*]' : [''],
    '[22*]' : ['[Cl]','[Br]','[I]'],
}

pattern = r'\[\d+\*\]'

def modified_smiles(smile):
    return re.sub(r'\[\d+\*\]', '[H]', smile)
class CompoundSplit():
    def __init__(self,args):
        self.args = args
        # self.search = True
        self.mol = Chem.MolFromSmiles(self.args.input)
        self.MW = Descriptors.MolWt(self.mol)

    def Filter_bricks(self,smile):
        mol = Chem.MolFromSmiles(smile)
        MW = Descriptors.MolWt(mol)
        if MW > 0.7 * self.MW:
            return False
        return True

    def Filter_bricks2(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return False  # 避免无效 SMILES 造成错误

        MW = Descriptors.MolWt(mol)  # 计算分子量
        heavy_atoms = mol.GetNumHeavyAtoms()  # 计算重原子数量
        original_heavy_atoms = self.mol.GetNumHeavyAtoms()  # 原分子的重原子数

        if MW > 0.7 * self.MW or heavy_atoms > original_heavy_atoms / 2:
            return False
        return True

    def get_leaves(self,recap_decomp, n=1):
        '''
        Recursively traverse the decomposition tree and save fragment SMILES to a text file
        '''
        for child in recap_decomp.children.values():
            with open('../../Data/3_output/bricap_decomp.txt', 'a') as f:
                f.write('\t' * n  + child.smiles + '\n')
            if child.children and n <= self.args.cut_num:  # Further inspect fragments
                self.get_leaves(child, n=n+1)

    def remove_duplicates(self, lst):
        '''
        Remove duplicate sublists based on shared elements
        '''
        seen = set()
        result = []
        for sublist in lst:
            sublist_frozenset = frozenset(sublist)
            # Check for duplicate elements between the current sublist and the processed sublists
            is_duplicate = False
            for seen_set in seen:
                if len(sublist_frozenset & seen_set) >= 2:
                    is_duplicate = True
                    break
            if not is_duplicate:
                seen.add(sublist_frozenset)
                result.append(sublist)
        return result

    def sort_block_scaffold(self,three_gen,three_gen_lst):
        three_gen_lst_sort = []
        for i in three_gen_lst:
            fra1, fra2, fra3 = i[0], i[1], i[2]
            frag1 = Chem.MolFromSmiles(modified_smiles(fra1))
            frag2 = Chem.MolFromSmiles(modified_smiles(fra2))
            frag3 = Chem.MolFromSmiles(modified_smiles(fra3))
            match1 = self.mol.GetSubstructMatches(frag1)
            match2 = self.mol.GetSubstructMatches(frag2)
            match3 = self.mol.GetSubstructMatches(frag3)

            if len(re.findall(pattern, fra1)) >= 2:
                three_gen['scaffold'].append(fra1)
                if match2 < match3:
                    three_gen['brick1'].append(fra2)
                    three_gen['brick2'].append(fra3)
                else:
                    three_gen['brick1'].append(fra3)
                    three_gen['brick2'].append(fra2)
                three_gen_lst_sort.append([fra2, fra3, fra1])
            elif len(re.findall(pattern, fra2)) >= 2:
                three_gen['scaffold'].append(fra2)
                if match1 < match3:
                    three_gen['brick1'].append(fra1)
                    three_gen['brick2'].append(fra3)
                else:
                    three_gen['brick1'].append(fra3)
                    three_gen['brick2'].append(fra1)
                three_gen_lst_sort.append([fra2, fra3, fra1])
            else:
                three_gen['scaffold'].append(fra3)
                if match2 < match3:
                    three_gen['brick1'].append(fra1)
                    three_gen['brick2'].append(fra2)
                else:
                    three_gen['brick1'].append(fra1)
                    three_gen['brick2'].append(fra2)
                three_gen_lst_sort.append([fra2, fra3, fra1])
        return three_gen,three_gen_lst_sort

    def decompose(self):
        self.recap_res = Bricap.RecapDecompose(self.mol, minFragmentSize=self.args.minFragmentSize,
                                               onlyUseReactions=self.args.onlyUseReactions)
        if self.args.tree:
            self.get_leaves(self.recap_res)
        childs = list(self.recap_res.children.keys())
        if self.args.cut_num == 1:
            two_gen = {'brick1':[], 'brick2':[]}
            two_gen_lst = []
            for i in range(0, len(childs), 2):
                frag1 = Chem.MolFromSmiles(modified_smiles(childs[i]))
                frag2 = Chem.MolFromSmiles(modified_smiles(childs[i + 1]))
                # 对打碎后的碎片进行五规则过滤
                if self.Filter_bricks(childs[i]) and self.Filter_bricks(childs[i + 1]):
                    # 匹配原子编号
                    match1 = self.mol.GetSubstructMatches(frag1)  # 获取 frag1 在原分子中的匹配原子编号
                    match2 = self.mol.GetSubstructMatches(frag2)  # 获取 frag2 在原分子中的匹配原子编号

                    if match1 and match2:
                        min_idx1 = min(match1[0])  # 取 frag1 最小的匹配原子编号
                        min_idx2 = min(match2[0])  # 取 frag2 最小的匹配原子编号

                        # 按编号大小排序
                        if min_idx1 < min_idx2:
                            two_gen['brick1'].append(childs[i])
                            two_gen['brick2'].append(childs[i + 1])
                            two_gen_lst.append([childs[i], childs[i + 1]])
                        else:
                            two_gen['brick1'].append(childs[i + 1])
                            two_gen['brick2'].append(childs[i])
                            two_gen_lst.append([childs[i + 1], childs[i]])
            print(two_gen)
            print(two_gen_lst)
            return two_gen_lst

        elif self.args.cut_num == 2:
            three_gen = {'brick1':[], 'scaffold':[], 'brick2':[]}
            three_gen_lst = []
            for index, child in enumerate(self.recap_res.children.values()):
                if child.children:
                    if index % 2 == 0 and self.Filter_bricks2(childs[index + 1]):
                        grandchildren = list(child.children.keys())
                        for i in range(0, len(grandchildren), 2):
                            if self.Filter_bricks2(grandchildren[i]) and self.Filter_bricks2(grandchildren[i + 1]):
                                three_gen_lst.append([grandchildren[i],grandchildren[i + 1],childs[index + 1]])

                    elif index % 2 != 0 and self.Filter_bricks2(childs[index - 1]):
                        grandchildren = list(child.children.keys())
                        for i in range(0, len(grandchildren), 2):
                            if self.Filter_bricks2(grandchildren[i]) and self.Filter_bricks2(grandchildren[i + 1]):
                                three_gen_lst.append([childs[index - 1], grandchildren[i],grandchildren[i + 1]])

            three_gen_lst = self.remove_duplicates(three_gen_lst)
            three_gen,three_gen_lst_sort = self.sort_block_scaffold(three_gen,three_gen_lst)
            # print(three_gen)
            # print(three_gen_lst_sort)
            return three_gen_lst_sort

def bricap_parse_args():
    parser = argparse.ArgumentParser(description='Chemical Compound Bricap')
    # parser.add_argument('-n', '--name', type=str, default="FGFR", help='compound name')
    parser.add_argument('-r', '--onlyUseReactions', type=list, default=None,
                        help='Fragmentation rules to be used')
    parser.add_argument('--tree', type=bool, default=True, help='Whether to output a tree diagram')
    args = parser.parse_args()
    return args

def Bricap_break(smiles, minFragmentSize,cut_num):
    bricap_args = bricap_parse_args()
    bricap_args.input = smiles
    bricap_args.minFragmentSize = minFragmentSize
    bricap_args.cut_num = cut_num
    ts = CompoundSplit(bricap_args)
    break_result_lst = ts.decompose()
    print("内置")
    print(break_result_lst)
    return break_result_lst

def Property_draw(data,current_time):
    mw, hac, slogp, hba, hbd, rotbonds, tpsa, qed, SAscore = data['MW'], data['HAC'], data['slogP'], data['HBA'], data['HBD'], data['RotBonds'], data['TPSA'], data['QED'], data['SAscore']
    fig, ax = plt.subplots(3, 3, figsize=(32, 16))  # kdeplot
    g1 = sns.kdeplot(mw, ax=ax[0][0], color='orange')
    ax[0][0].hist(mw, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[0][0].set_xlabel("Molecular weight")

    g2 = sns.kdeplot(hac, ax=ax[0][1], color='orange')
    ax[0][1].hist(hac, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[0][1].set_xlabel("Hac")

    g3 = sns.kdeplot(slogp, ax=ax[0][2], color='orange')
    ax[0][2].hist(slogp, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[0][2].set_xlabel("slogp")

    g4 = sns.kdeplot(hba, ax=ax[1][0], color='orange')
    ax[1][0].hist(hba, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[1][0].set_xlabel("Hbond acceptor")

    g5 = sns.kdeplot(hbd, ax=ax[1][1], color='orange')
    ax[1][1].hist(hbd, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[1][1].set_xlabel("Hbond donor")

    g6 = sns.kdeplot(rotbonds, ax=ax[1][2], color='orange')
    ax[1][2].hist(rotbonds, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[1][2].set_xlabel("Rotational bond")

    g7 = sns.kdeplot(tpsa, ax=ax[2][0], color='orange')
    ax[2][0].hist(tpsa, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[2][0].set_xlabel("TPSA")

    g8 = sns.kdeplot(qed, ax=ax[2][1], color='orange')
    ax[2][1].hist(qed, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[2][1].set_xlabel("QED")

    g9 = sns.kdeplot(SAscore, ax=ax[2][2], color='orange')
    ax[2][2].hist(SAscore, bins=20, color='lightblue', alpha=0.7, density=True, label='Histogram')
    ax[2][2].set_xlabel("SAscore")

    for ax in plt.gcf().axes:
        l = ax.get_xlabel()
        ll = ax.get_ylabel()
        ax.set_xlabel(l, fontsize=20)
        ax.set_ylabel(ll, fontsize=20)
    plt.savefig(f'../../scripts/plot/properties_{current_time}.png', dpi=1000, bbox_inches='tight')
    return plt

def cal_mol_props(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if not m:
            return None, None, None, None, None, None, None, None, None
        mw = Descriptors.MolWt(m)
        hac = Descriptors.HeavyAtomCount(m)
        slogp = Descriptors.MolLogP(m)
        hba = Descriptors.NumHAcceptors(m)
        hbd = Descriptors.NumHDonors(m)
        rob = Descriptors.NumRotatableBonds(m)
        tpsa = Descriptors.TPSA(m)
        qed = QED.qed(m)
        SAscore = sascorer.calculateScore(m)
        # chiral_center = len(Chem.FindMolChiralCenters(m))
        # aRings = rdMolDescriptors.CalcNumAromaticRings(m)
        return mw, hac, slogp, hba, hbd, rob, tpsa, qed, SAscore
    except Exception as e:
        print(e)
        return None, None, None, None, None, None, None, None, None
def search_parse_args():
    parser = argparse.ArgumentParser(description='Chemical Space Search')
    parser.add_argument('-p', '--property', type=bool, default=True, help='Whether to output brick properties')
    parser.add_argument('-o', '--output', type=str, default='../../Data/3_output/bricap_result',
                        help='Output result after brick assembly')
    args = parser.parse_args()
    return args

def Chemical_Space_Search(smiles_input,BB_data,fingerprint,Endpoint_threshold
                          ,Scaffold_threshold,break_list,chemprop_checkboxgroup):

# def Chemical_Space_Search(smiles_input, BB_data, fingerprint, Endpoint_threshold
#                           , Scaffold_threshold, break_list, chemprop_checkboxgroup,name):

    '''
    解决问题checkbox为空怎么办
    '''

    start_time = time.time()
    search_args = search_parse_args()
    search_args.input = smiles_input
    search_args.BB_data = BB_data
    search_args.fingerprint = fingerprint
    if isinstance(break_list, str):
        break_list = ast.literal_eval(break_list)
    search_args.cut_num = len(break_list[0]) - 1
    search_args.Endpoint_threshold,search_args.Scaffold_threshold = Endpoint_threshold,Scaffold_threshold
    search_args.break_list = break_list
    search_args.cal_property = chemprop_checkboxgroup

    # Determine the length of breakage based on the length of the result list
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    search_args.output_path = os.path.join(search_args.output
            , f'{search_args.cut_num}_{search_args.fingerprint}_{current_time}.csv')

    us = Universe_search(search_args)
    print("初始化消耗时间：",time.time()-start_time)
    replace_result, search_result, dfs = us.Replace_dummys(break_list)
    product_df = us.get_result(break_list, dfs)
    print(product_df)
    plt = Property_draw(product_df,current_time)
    return product_df, plt

    # '''这好像是cdk9后面干啥用的'''
    # # 写一个csv文件，第一列是break_list的各种组合，第二列是替换后的各种组合，第三列是搜索结果
    # break_column = []
    # for i in break_list:
    #     break_column.extend(i)
    # replace_column = []
    # for j in break_column:
    #     replace_column.append(replace_result[j])
    # search_column = []
    # for k in break_column:
    #     search_column.append(search_result[k])
    #
    # # 让每三行后插入一个空行
    # new_break_column = []
    # new_replace_column = []
    # new_search_column = []
    #
    # for i in range(len(break_column)):
    #     new_break_column.append(break_column[i])
    #     new_replace_column.append(replace_column[i])
    #     new_search_column.append(search_column[i])
    #
    #     # 每三行后插入一个空行
    #     if (i + 1) % 3 == 0:  # 注意这里是从0开始索引
    #         new_break_column.append('')
    #         new_replace_column.append('')
    #         new_search_column.append('')
    #
    # search_args.break_output_path = f'CDK9/{name}_{search_args.Endpoint_threshold}_{search_args.Scaffold_threshold}.csv'
    # data = pd.DataFrame({'break_list':new_break_column,'replace_result':new_replace_column,'search_result':new_search_column})
    # data.to_csv(search_args.break_output_path,index=False)


class Universe_search:
    def __init__(self,args):
        self.args = args
        self.mol = Chem.MolFromSmiles(self.args.input)
        self.ap = assemle_products()
        self.initialize_BB_data()
        self.initialize_fps()

    def Filter_products(self,smile):
        mol = Chem.MolFromSmiles(smile)
        MW = Descriptors.MolWt(mol)
        HBA = Descriptors.NOCount(mol)
        HBD = Descriptors.NHOHCount(mol)
        LogP = Descriptors.MolLogP(mol)
        rob = Descriptors.NumRotatableBonds(mol)
        conditions = [MW <= 600, HBA <= 10, HBD <= 5, LogP <= 5, rob <= 10]
        if conditions.count(True) >= 5:
            return True
        else:
            return False

    def initialize_BB_data(self):
        if self.args.BB_data == 'Enamine BB Catalog':
            file_path = '../../Data/2_rules/enamine_bbcat_to_rules.feather'
        elif self.args.BB_data == "Enamine stock":
            file_path = '../../Data/2_rules/enamine_stock_to_rules.feather'
        else:
            file_path = '../../Data/2_rules/mcule_to_rules.feather'
        # with open(file_path, 'rb') as file:
        #     self.bricks_table = pickle.load(file)
        self.bricks_table = pd.read_feather(file_path)

    def initialize_fps(self):
        # 根据不同砌块库和指纹类型选择相应文件
        fingerprint_files = {
            ('Enamine BB Catalog', 'MACCS'): '../../Data/2_rules/enamine_bbcat_maccs_fp.pkl',
            ('Enamine BB Catalog', 'Morgan'): '../../Data/2_rules/enamine_bbcat_morgan_fp.pkl',
            # ('Enamine BB Catalog', 'Morgan'): '../../Data/2_rules/total_brick_morgan_fps.pkl',
            ('Enamine stock', 'MACCS'): '../../Data/2_rules/enamine_stock_maccs_fp.pkl',
            ('Enamine stock', 'Morgan'): '../../Data/2_rules/enamine_stock_morgan_fp.pkl',
            ('mcule', 'MACCS'): '../../Data/2_rules/mcule_maccs_fp.pkl',
            ('mcule', 'Morgan'): '../../Data/2_rules/mcule_morgan_fp.pkl'
        }
        # 获取当前砌块库和指纹类型
        key = (self.args.BB_data, self.args.fingerprint)

        # 检查文件是否存在于字典中
        if key not in fingerprint_files:
            raise ValueError(
                f"Invalid combination of BB_data '{self.args.BB_data}' and fingerprint '{self.args.fingerprint}'")

        # 加载指纹文件
        self.fps_store = fingerprint_files[key]
        with open(self.fps_store, 'rb') as f:
            self.fps = pickle.load(f)

    # def load_molecules(self):
    #     self.molecules = self.bricks_table['MOL']
    #     return self.calculate_fps()
    #
    # def calculate_fps(self):
    #     if self.molecules is not None and not self.molecules.empty:
    #         fingerprint_function = MACCSkeys.GenMACCSKeys if self.fingerprint == 'maccs' else lambda \
    #                 x: AllChem.GetMorganFingerprintAsBitVect(x, self.radius, self.nbits)
    #         self.fps = [fingerprint_function(x) for x in self.molecules]
    #     return self.fps
    #
    # def save_fps(self, output_file):
    #     with open(output_file, 'wb') as f:
    #         pickle.dump(self.fps, f)
    #     print(f'File saved: {output_file}')

    def brick_dfs(self, lst):
        '''
        Returns a dataframe of bricks and their reaction rules based on input IDs
        '''
        dfs = self.bricks_table.loc[self.bricks_table['ID'].isin(lst)]
        return dfs

    def calculate_props(self, df):
        start_time = time.time()
        df['MOL'] = df['Product'].apply(Chem.MolFromSmiles)
        self_mol_fp = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, 1024)
        df['Fingerprint'] = df['MOL'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2,1024))
        df['Similarity'] = df['Fingerprint'].apply(lambda x: DataStructs.FingerprintSimilarity(self_mol_fp, x))
        # 初始化属性计算字典
        property_calculators = {
            'MW': lambda x: Descriptors.MolWt(x),
            'HAC': lambda x: Descriptors.NumHAcceptors(x),
            'slogP': lambda x: Descriptors.MolLogP(x),
            'HBA': lambda x: Descriptors.NumHAcceptors(x),
            'HBD': lambda x: Descriptors.NumHDonors(x),
            'TPSA': lambda x: Descriptors.TPSA(x),
            'RotBonds': lambda x: Descriptors.NumRotatableBonds(x),
            'QED': lambda x: QED.qed(x),
            'SAscore': lambda x: sascorer.calculateScore(x)
        }

        # 根据选择的属性进行计算
        for prop in self.args.cal_property:
            if prop in property_calculators:
                df[prop] = df['MOL'].apply(property_calculators[prop])

        # Remove the 'MOL' column and sort by similarity in descending order
        df.drop(columns=['MOL','Fingerprint'], inplace=True)
        df.sort_values(by='Similarity', ascending=False, inplace=True)
        print(f'计算属性耗费时间：{time.time() - start_time}')
        return df

    def cal_mol_props(self,smi):
        try:
            m = Chem.MolFromSmiles(smi)
            if not m:
                return None, None, None, None, None, None, None, None, None
            mw = Descriptors.MolWt(m)
            hac = Descriptors.HeavyAtomCount(m)
            slogp = Descriptors.MolLogP(m)
            hba = Descriptors.NumHAcceptors(m)
            hbd = Descriptors.NumHDonors(m)
            rob = Descriptors.NumRotatableBonds(m)
            tpsa = Descriptors.TPSA(m)
            qed = QED.qed(m)
            SAscore = sascorer.calculateScore(m)
            # chiral_center = len(Chem.FindMolChiralCenters(m))
            # aRings = rdMolDescriptors.CalcNumAromaticRings(m)
            return mw, hac, slogp, hba, hbd, rob, tpsa, qed, SAscore
        except Exception as e:
            print(e)
            return None, None, None, None, None, None, None, None, None


    def calculate_props_linux(self, df):
        start_time = time.time()
        pandarallel.initialize(nb_workers=12, progress_bar=True)
        df['MW'], df['HAC'], df['slogP'], df['HBA'], df['HBD'], df['RotBonds'], df['TPSA'], df['QED'], df['SAscore'] \
            = zip(*df['SMILES'].parallel_apply(self.cal_mol_props))
        # Remove the 'MOL' column and sort by similarity in descending order
        df.drop(columns=['MOL', 'Fingerprint'], inplace=True)
        df.sort_values(by='Similarity', ascending=False, inplace=True)
        print(f'linux下计算属性耗费时间：{time.time() - start_time}')
        return df


    def Replace_dummys(self,initial_results):
        '''
        Replace dummy atoms in the fragments with environment atoms
        :param initial_results:
        :return:
        '''

        start_time = time.time()
        deduplication_set = set(chain.from_iterable(initial_results))
        replace_dict = {key: [] for key in deduplication_set}
        # Replace atoms at breaking points
        search_dict = {key: [] for key in deduplication_set}
        # print(deduplication_set)
        for fragment in deduplication_set:
            pattern_match = re.findall(pattern, fragment)
            if len(pattern_match) >= 2:
                fragments1 = []
                fragments2 = []
                for index, dummy in enumerate(pattern_match):
                    if dummy == '[13*]':
                        position = fragment.find('[13*]')
                        if position == 0:  # [13*] 在字符串开头
                            replacement = environs[dummy][1]
                        else:
                            replacement = environs[dummy][0]
                    else:
                        replacement = environs[dummy]
                    if index == 0:
                        for replace in replacement:
                            fragments1.append(fragment.replace(dummy, replace, 1).replace('()', ''))
                    else:
                        for inter_fragment in fragments1:
                            for replace in replacement:
                                fragments2.append(inter_fragment.replace(dummy, replace, 1).replace('()', ''))
                        replace_dict[fragment] = fragments2
            else:
                for dummy in pattern_match:
                    if dummy == '[13*]':
                        position = fragment.find('[13*]')
                        if position == 0:  # [13*] 在字符串开头
                            replacement = environs[dummy][1]
                        else:
                            replacement = environs[dummy][0]
                    else:
                        replacement = environs[dummy]
                    fragments = []
                    # replacement = environs[dummy]
                    for i in replacement:
                        fragments.append(fragment.replace(dummy, i, 1).replace('()', ''))
                replace_dict[fragment] = fragments

        print("替换耗费时间：",time.time()-start_time)

        start_time = time.time()
        for key, values in tqdm(replace_dict.items(), desc="Processing keys in replace_dict"):
            try:
                scffold_flag = len(re.findall(pattern, key)) >= 2
                for value in values:
                    sim_list = self.search_bricks(value, scffold_flag)
                    if len(sim_list) > 0:
                        search_dict[key].extend(sim_list)
            except Exception as e:
                print(value)
                print(e)
                continue

        # Deduplicate the values for each key in search_dict
        for key, value in search_dict.items():
            search_dict[key] = list(set(value))

        print(search_dict)
        dfs = {key: self.brick_dfs(value) for key, value in search_dict.items()}


        print("搜索耗费时间：", time.time() - start_time)
        return replace_dict, search_dict, dfs

    def search_bricks(self, smile, scaffold_flag):
        if scaffold_flag:
            threshold = self.args.Scaffold_threshold
        else:
            threshold = self.args.Endpoint_threshold
        # Performs similarity search for a given fragment SMILES
        m = Chem.MolFromSmiles(smile)
        # Choose fingerprint type and file path based on MACCS flag
        if self.args.fingerprint == 'MACCS':
            m_fps = MACCSkeys.GenMACCSKeys(m)
        elif self.args.fingerprint == 'Morgan':
            m_fps = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
        elif self.args.fingerprint == 'Pharm2D':
            m_fps = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
        else:
            raise ValueError('Invalid fingerprint type')
        similarities = DataStructs.BulkTanimotoSimilarity(m_fps, self.fps)
        similarities_np = np.array(similarities)
        similar_indices = np.where(similarities_np > threshold)[0]

        ids = self.bricks_table['ID'].tolist()
        sim_list = np.array(ids)[similar_indices]
        return sim_list

    def get_result(self,initial_results,dfs):
        # Decompose and search for fragments
        if self.args.cut_num == 1:
            start_time = time.time()
            two_gen_df = pd.DataFrame(columns=['Reactant1', 'Reactant1_ID', 'Reactant2', 'Reactant2_ID', 'Product'])
            for i, result in enumerate(initial_results):
                matrix = [dfs[key] for key in result]
                # Skip if any matrix is empty
                if any(df.empty for df in matrix):
                    continue
                dfs[result[0]].loc[:, 'MOL'] = dfs[result[0]]['SMILES'].apply(Chem.MolFromSmiles)
                dfs[result[1]].loc[:, 'MOL'] = dfs[result[1]]['SMILES'].apply(Chem.MolFromSmiles)
                result_dict = self.ap.match_bricks(matrix)
                products = self.ap.trans_bricks(result_dict, matrix)
                if products.empty:
                    continue
                # products.drop_duplicates(
                #     subset=['Reactant1', 'Reactant1_ID', 'Reactant2', 'Reactant2_ID', 'Product'], keep='first',
                #     inplace=True)
                products.drop_duplicates(
                        subset=['Product'], keep='first',
                        inplace=True)
                products['Lipinski'] = products['Product'].apply(self.Filter_products)
                products_filtered = products[products['Lipinski']].drop(columns=['Lipinski'])
                products_filtered['ID'] = range(len(products_filtered))

                two_gen_df = pd.concat([two_gen_df.reset_index(drop=True), products_filtered.reset_index(drop=True)],
                                         ignore_index=True)
                print(f'The {i}-th breaking result and Generated three-component product length: {len(two_gen_df)}')
                print(f'第{i}次打碎结果生成产物所花时间：{time.time() - start_time}')

                # two_gen_df = pd.concat([two_gen_df, pd.DataFrame(products)], ignore_index=True)

            # Save results
            print(two_gen_df)
            df_result = self.calculate_props(two_gen_df)
            start_time = time.time()
            df_result.to_csv(self.args.output_path, index=False)
            print(f'Time taken: {time.time() - start_time}')
            return df_result
        else:
            three_gen_df = pd.DataFrame(
                columns=['Reactant1', 'Reactant1_ID', 'Scaffold', 'Scaffold_ID', 'Reactant2', 'Reactant2_ID', 'Product'])
            for i, result in enumerate(initial_results):
                start_time = time.time()
                matrix = [dfs[key] for key in result]
                # Skip if any matrix is empty
                if any(df.empty for df in matrix):
                    continue
                dfs[result[0]]['MOL'] = dfs[result[0]]['SMILES'].apply(Chem.MolFromSmiles)
                dfs[result[1]]['MOL'] = dfs[result[1]]['SMILES'].apply(Chem.MolFromSmiles)
                matrix1 = [dfs[result[0]]] + [dfs[result[1]]]
                result_dict = self.ap.match_bricks(matrix1)
                temp_products = self.ap.trans_bricks(result_dict, matrix1)
                if temp_products.empty:
                    continue
                # Remove duplicates
                # temp_products.drop_duplicates(
                #     subset=['Reactant1', 'Reactant1_ID', 'Reactant2', 'Reactant2_ID', 'Product'], keep='first',
                #     inplace=True)
                temp_products.drop_duplicates(
                    subset=['Product'], keep='first',
                    inplace=True)
                temp_products['Lipinski'] = temp_products['Product'].apply(self.Filter_products)
                temp_filtered = temp_products[temp_products['Lipinski']].drop(columns=['Lipinski'])
                temp_filtered['ID'] = range(len(temp_filtered))
                temp_filtered = self.brick_to_rule(temp_filtered)

                # Rename 'Product' column to 'SMILES'
                temp_filtered.rename(columns={'Product': 'SMILES'}, inplace=True)

                dfs[result[-1]]['MOL'] = dfs[result[-1]]['SMILES'].apply(Chem.MolFromSmiles)
                matrix2 = [temp_filtered.iloc[:, 4:]] + [dfs[result[-1]]]
                result2_dict = self.ap.match_bricks(matrix2)
                terminal_products = self.ap.trans_bricks(result2_dict, matrix2)

                if terminal_products.empty:
                    continue
                # terminal_products.drop_duplicates(
                #     subset=['Reactant1', 'Reactant1_ID', 'Reactant2', 'Reactant2_ID', 'Product'], keep='first',
                #     inplace=True)
                terminal_products.drop_duplicates(
                    subset=['Product'], keep='first',
                    inplace=True)
                terminal_products['Lipinski'] = terminal_products['Product'].apply(self.Filter_products)
                # Filter rows that satisfy Lipinski's Rule of Five
                terminal_filtered = terminal_products[terminal_products['Lipinski']].drop(columns=['Lipinski'])

                # Populate three_gen_df with filtered results
                terminal_filtered = terminal_filtered.rename(columns={'Reactant1': 'temp_production', 'Reactant1_ID': 'temp_ID'})
                temp_filtered = temp_filtered.iloc[:, :6].rename(columns={'Reactant2': 'Scaffold', 'Reactant2_ID': 'Scaffold_ID', 'SMILES':'temp_production', 'ID':'temp_ID'})
                merged_df = pd.merge(temp_filtered, terminal_filtered, on=['temp_production', 'temp_ID'], how='right')
                merged_df.drop(columns=['temp_production', 'temp_ID'], inplace=True)
                three_gen_df = pd.concat([three_gen_df.reset_index(drop=True), merged_df.reset_index(drop=True)], ignore_index=True)

                print(f'The {i}-th breaking result and Generated three-component product length: {len(three_gen_df)}')
                print(f'第{i}次打碎结果生成产物所花时间：{time.time() - start_time}')
            # Output final result
            # print(three_gen_df)

            three_gen_df = self.calculate_props(three_gen_df)
            start_time = time.time()
            three_gen_df.to_csv(self.args.output_path, index=False)
            print(f'保存耗费时间：{time.time() - start_time}')
            return three_gen_df

    def brick_to_rule(self, result_df):
        rules_df = pd.read_excel('../../Data/2_rules/rules.xlsx')
        reactions = ['1,2,4-triazole_carboxylic-acid_ester', 'Amidecoupling', 'Benzimidazole_derivatives_aldehyde'
            , 'Benzimidazo le_derivatives_carboxylic-acid_ester', 'Benzofuran', 'Benzothiazole'
            , 'Benzothiophene', 'Benzoxazole_arom-aldehyde', 'Benzoxazole_carboxylic-acid', 'Buchwald-Hartwig'
            , 'Decarboxylative_coupling', 'Fischer indole', 'Friedlaender_quinoline', 'Grignard_alcohol'
            , 'Grignard_carbonyl', 'Heck_non-terminal_vinyl', 'Heck_terminal_vinyl', 'Heteroaromatic_nuc_sub'
            , 'Hinsberg', 'Huisgen_Cu-catalyzed_1,4-subst', 'Huisgen_Ru-catalyzed_1,5_subst', 'Indole'
            , 'Mitsunobu_imide', 'Mitsunobu_phenol', 'Mitsunobu_sulfonamide', 'N-arylation_heterocycles', 'Negishi'
            , 'Niementowski_quinazoline', 'Nucl_sub_aromatic_ortho_nitro', 'Nucl_sub_aromatic_para_nitro',
                     'Oxadiazole'
            , 'Phthalazinone', 'Pictet-Spengler', 'Piperidine_indole', 'Reductive amination', 'Sonogashira'
            , 'Spirochromanone', 'Stille', 'Suzuki', 'Thiazole', 'Thiourea', 'Urea', 'Williamson_ether']
        result_df['MOL'] = result_df['Product'].apply(Chem.MolFromSmiles)
        for index, reaction in enumerate(reactions):
            try:
                smirks_series = rules_df.loc[rules_df['Name_reactions'] == reaction, 'Smirks'].iloc[0]
                react1 = smirks_series.split('>>')[0].split('.')[0]
                react2 = smirks_series.split('>>')[0].split('.')[1]
                result_df[reactions[index]] = result_df['MOL'].apply(
                    lambda mol: self.determine_reaction_type(mol, react1, react2))
            except Exception as e:
                print(e)
                print(reaction)
                continue
        mol_column = result_df.pop('MOL')
        result_df['MOL'] = mol_column
        return result_df

    def determine_reaction_type(self, mol, reactant1_smarts, reactant2_smarts):
        """
        Determines the type of reaction based on substructure matches in the molecule.

        :param mol: The molecule to check.
        :param reactant1_smarts: SMARTS pattern for the first reactant.
        :param reactant2_smarts: SMARTS pattern for the second reactant.
        :return:
            - 3 if the molecule matches both reactant1 and reactant2.
            - 1 if the molecule matches only reactant1.
            - 2 if the molecule matches only reactant2.
            - 0 if the molecule matches neither reactant1 nor reactant2.
        """
        react1_match = mol.HasSubstructMatch(Chem.MolFromSmarts(reactant1_smarts))
        react2_match = mol.HasSubstructMatch(Chem.MolFromSmarts(reactant2_smarts))
        if react1_match and react2_match:
            return 3
        elif react1_match:
            return 1
        elif react2_match:
            return 2
        else:
            return 0

class assemle_products():
    def __init__(self,excel_path='../../Data/2_rules/rules.xlsx',property_flag=True,three_gen=False):
        self.df = pd.read_excel(excel_path)
        self.property = property_flag
        self.three_gen = three_gen

    def match_bricks(self,matrix):
        '''
        Match the bricks to the rules
        :param results:
        :return:
        '''
        columns =['1,2,4-triazole_carboxylic-acid_ester', 'Amidecoupling', 'Benzimidazole_derivatives_aldehyde'
            , 'Benzimidazole_derivatives_carboxylic-acid_ester', 'Benzofuran', 'Benzothiazole'
            , 'Benzothiophene', 'Benzoxazole_arom-aldehyde', 'Benzoxazole_carboxylic-acid', 'Buchwald-Hartwig'
            , 'Decarboxylative_coupling', 'Fischer indole', 'Friedlaender_quinoline', 'Grignard_alcohol'
            , 'Grignard_carbonyl', 'Heck_non-terminal_vinyl', 'Heck_terminal_vinyl', 'Heteroaromatic_nuc_sub'
            , 'Hinsberg', 'Huisgen_Cu-catalyzed_1,4-subst', 'Huisgen_Ru-catalyzed_1,5_subst', 'Indole'
            , 'Mitsunobu_imide', 'Mitsunobu_phenol', 'Mitsunobu_sulfonamide', 'N-arylation_heterocycles', 'Negishi'
            , 'Niementowski_quinazoline', 'Nucl_sub_aromatic_ortho_nitro', 'Nucl_sub_aromatic_para_nitro', 'Oxadiazole'
            , 'Phthalazinone', 'Pictet-Spengler', 'Piperidine_indole', 'Reductive amination', 'Sonogashira'
            , 'Spirochromanone', 'Stille', 'Suzuki', 'Thiazole', 'Thiourea', 'Urea', 'Williamson_ether']
        results_dict = {'brick1':[], 'brick2':[], 'rule':[], 'brick1_value':[], 'brick2_value':[]}
        brick1 = matrix[0].iloc[:,2:-1]
        brick2 = matrix[1].iloc[:,2:-1]
        array1 = brick1.values
        array2 = brick2.values

        row1, col1 = array1.shape

        # Perform matrix comparison
        for k in range(col1):
            # 比较 array1 和 array2 的第 k 列
            col1_values = array1[:, k][:, np.newaxis]  # (row1, 1)
            col2_values = array2[:, k]  # (row2,)

            # 利用广播机制和条件筛选
            result_matrix = (col1_values != 0) & (col2_values != 0) & (col1_values != col2_values)

            # 仅在有匹配时提取结果
            rows, cols = np.where(result_matrix)
            if rows.size > 0:
                # 预取 ID 和规则，减少多次访问
                brick1_ids = matrix[0].iloc[rows]['ID'].to_list()
                brick2_ids = matrix[1].iloc[cols]['ID'].to_list()

                # 将匹配结果直接添加到字典
                results_dict['brick1'].extend(brick1_ids)
                results_dict['brick2'].extend(brick2_ids)
                results_dict['rule'].extend([columns[k]] * len(rows))
                results_dict['brick1_value'].extend(array1[rows, k])
                results_dict['brick2_value'].extend(array2[cols, k])

        return results_dict

    def add_to_chemical_space(self,chemical_space, filter_row1, filter_row2, product):
        '''
        Add the reactants and products to the chemical space
        :param chemical_space: dictionary to store chemical space
        :param filter_row1: filtered row 1
        :param filter_row2: filtered row 2
        :param product: resulting product
        '''
        chemical_space['Reactant1'].append(filter_row1.iloc[0]['SMILES'])
        chemical_space['Reactant1_ID'].append(filter_row1.iloc[0]['ID'])
        chemical_space['Reactant2'].append(filter_row2.iloc[0]['SMILES'])
        chemical_space['Reactant2_ID'].append(filter_row2.iloc[0]['ID'])
        chemical_space['Product'].append(product)

    def reaction(self,matrix,index1,index2,rule,brick1_value,brick2_value):
        '''
        Perform chemical reactions based on input bricks and rules
        :param matrix: input matrix with reactants
        :param index1: index of brick1
        :param index2: index of brick2
        :param rule: reaction rule
        :param brick1_value: value associated with brick1
        :param brick2_value: value associated with brick2
        :return: DataFrame of products
        '''
        filter_row1 = matrix[0].loc[matrix[0].loc[:, 'ID'] == index1]
        filter_row2 = matrix[1].loc[matrix[1].loc[:, 'ID'] == index2]
        mol1 = filter_row1['MOL'].values[0]
        mol2 = filter_row2['MOL'].values[0]

        chemical_space = {'Reactant1': [], 'Reactant1_ID': [], 'Reactant2': [], 'Reactant2_ID': [], 'Product': []}
        rxn = AllChem.ReactionFromSmarts(self.df.loc[self.df.loc[:, 'Name_reactions'] == rule, 'Smirks'].iloc[0])
        if brick1_value == 1 and brick2_value == 2:
            reaction_result = rxn.RunReactants((mol1, mol2))
            for i in range(len(reaction_result)):
                mol = reaction_result[i][0]
                product = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False, allHsExplicit=False)
                self.add_to_chemical_space(chemical_space, filter_row1, filter_row2, product)
        elif brick1_value == 2 and brick2_value == 1:
            reaction_result = rxn.RunReactants((mol2, mol1))
            for i in range(len(reaction_result)):
                mol = reaction_result[i][0]
                product = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False, allHsExplicit=False)
                self.add_to_chemical_space(chemical_space, filter_row1, filter_row2, product)
        elif brick1_value == 3:
            reaction_result1 = rxn.RunReactants((mol1, mol2))
            reaction_result2 = rxn.RunReactants((mol2, mol1))
            reaction_result = reaction_result1 + reaction_result2
            for i in range(len(reaction_result)):
                mol = reaction_result[i][0]
                product = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False, allHsExplicit=False)
                self.add_to_chemical_space(chemical_space, filter_row1, filter_row2, product)
        return pd.DataFrame(chemical_space)

    def trans_bricks(self,results_dict,matrix):
        '''
        Transform the bricks into products using chemical rules
        :param results_dict: dictionary with matched bricks and rules
        :param matrix: input matrix containing reactants
        :return: DataFrame with final products
        '''
        exectuor = ThreadPoolExecutor(max_workers=os.cpu_count())
        final_results = pd.DataFrame(columns=['Reactant1', 'Reactant1_ID', 'Reactant2', 'Reactant2_ID', 'Product'])
        futures = []
        start_time = time.time()
        for i in range(len(results_dict['brick1'])):
            future = exectuor.submit(self.reaction, matrix,results_dict['brick1'][i], results_dict['brick2'][i]
                                         , results_dict['rule'][i], results_dict['brick1_value'][i]
                                         , results_dict['brick2_value'][i])
            futures.append(future)

        with tqdm(total=len(futures), desc='Processing') as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    final_results = pd.concat([final_results,result], ignore_index=True)
                except Exception as e:
                    print(e)
                    continue
                pbar.update(1)
        print(final_results)
        print(f'组装time: {time.time() - start_time}')
        return final_results

if __name__ == '__main__':
    break_list = Bricap_break('FC(C=C1)=CC(OC)=C1C2=CC(NC3=CN(C4CCNCC4)N=C3)=NC=N2'
                              ,4,2)

    # break_list = Bricap_break('ClC1=CN=C(CC2=CN(C3CCNCC3)N=C2)C=C1C4=CN(C5CC5)N=C4'
    #                           , 4, 2)

    Chemical_Space_Search('FC(C=C1)=CC(OC)=C1C2=CC(NC3=CN(C4CCNCC4)N=C3)=NC=N2',"Enamine BB Catalog", "Morgan", 0.6, 0.6
                 , break_list, ['MW', 'HAC','slogP','HBA','HBD','TPSA','RotBonds','QED','SAscore'])

    # with open('../../cdk9Ⅰ.txt', 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    # for index,line in enumerate(lines):
    #     # if index != 13:
    #     #     continue
    #     line = line.strip()
    #     smile = line.split(' ')[0]
    #     name = line.split(' ')[1]
    #     break_list = Bricap_break(smile, 3, 2)
    #     Chemical_Space_Search(smile, "Enamine BB Catalog", "Morgan", 0.8, 0.8
    #                           , break_list, ['MW', 'HAC', 'slogP', 'HBA', 'HBD', 'TPSA', 'RotBonds'
    #                               , 'QED', 'SAscore'], name)
