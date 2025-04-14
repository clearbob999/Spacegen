import argparse
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap, AllChem, DataStructs, Descriptors, MACCSkeys, QED
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from tqdm import tqdm
from scripts import Bricap
from itertools import chain
import time
import re
from utils import *
from rdkit.Contrib.SA_Score import sascorer


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
    '[13*]' : [['=[O]'],['[O]=']],
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

class CompoundSplit():
    def __init__(self,args):
        self.args = args
        self.mol = Chem.MolFromSmiles(self.args.input)
        self.MW = Descriptors.MolWt(self.mol)
        self.initialize_BB_data()
        self.initialize_fps()
        # self.environs = {
        #     '[1*]': ['[OH]'],
        #     '[2*]': [''],
        #     '[3*]': ['[OH]'],
        #     '[4*]': ['[OH]'],
        #     '[5*]': ['[H]'],
        #     '[6*]': ['[Cl]'],
        #     '[7*]': ['[Cl]'],
        #     '[8*]': [''],
        #     '[9*]': ['[O]=[C]='],
        #     '[10*]': [''],
        #     '[11*]': ['[OH]'],
        #     '[12*]': ['[Cl]'],
        #     '[13*]': ['[O]='],
        #     '[14*]': ['[Cl]'],
        #     '[15*]': [''],
        #     '[16*]': ['[Cl]'],
        #     '[17*]': [''],
        #     '[18*]': ['[Cl]'],
        #     '[19*]': ['OB(O)'],
        #     '[20*]': ['[Cl]', '[Br]', '[I]'],
        #     '[21*]': [''],
        #     '[22*]': ['[Cl]', '[Br]', '[I]'],
        # }
        self.pattern = r'\[\d+\*\]'

    def initialize_BB_data(self):
        if self.args.BB_data == 'Enamine BB Catalog':
            file_path = '../../Data/2_rules/enamine_bbcat_to_rules.feather'
        elif self.args.BB_data == "Enamine stock":
            file_path = '../../Data/2_rules/enamine_stock_to_rules.feather'
        else:
            file_path = '../../Data/2_rules/mcule_to_rules.feather'
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

    def brick_dfs(self, lst):
        '''
        Returns a dataframe of bricks and their reaction rules based on input IDs
        '''
        dfs = self.bricks_table.loc[self.bricks_table['ID'].isin(lst)]
        return dfs

    def Filter_bricks(self, smile):
        mol = Chem.MolFromSmiles(smile)
        MW = Descriptors.MolWt(mol)
        if MW > 0.8 * self.MW:
            return False
        return True

    def get_leaves(self,recap_decomp, n=1, fragments_list=None):
        '''
        Recursively traverse the decomposition tree and save fragment SMILES to a text file
        '''
        if fragments_list is None:
            fragments_list = []
        for child in recap_decomp.children.values():
            fragments_list.append(child.smiles)
            # with open('../../Data/3_output/bricap_decomp.txt', 'a') as f:
            #     f.write('\t' * n  + child.smiles + '\n')
            if child.children and n <= self.args.cut_num:  # Further inspect fragments
                self.get_leaves(child, n=n+1, fragments_list=None)
        return fragments_list


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
            if len(re.findall(self.pattern, fra1)) >= 2:
                three_gen['scaffold'].append(fra1)
                three_gen['brick1'].append(fra2)
                three_gen['brick2'].append(fra3)
                three_gen_lst_sort.append([fra2,fra1,fra3])
            elif len(re.findall(self.pattern, fra2)) >= 2:
                three_gen['scaffold'].append(fra2)
                three_gen['brick1'].append(fra1)
                three_gen['brick2'].append(fra3)
                three_gen_lst_sort.append([fra1, fra2, fra3])
            else:
                three_gen['scaffold'].append(fra3)
                three_gen['brick1'].append(fra2)
                three_gen['brick2'].append(fra1)
                three_gen_lst_sort.append([fra1, fra3, fra2])
        return three_gen,three_gen_lst_sort


    def decompose(self,mol=None):
        if mol is None:
            self.recap_res = Bricap.RecapDecompose(self.mol, minFragmentSize=self.args.minFragmentSize,
                                               onlyUseReactions=self.args.onlyUseReactions)
        else:
            self.recap_res = Bricap.RecapDecompose(mol, minFragmentSize=self.args.minFragmentSize,
                                               onlyUseReactions=self.args.onlyUseReactions)

        childs = list(self.recap_res.children.keys())
        if self.args.cut_num == 1:
            two_gen = {'brick1': [], 'brick2': []}
            two_gen_lst = []
            for i in range(0, len(childs), 2):
                # 对打碎后的碎片进行五规则过滤
                if self.Filter_bricks(childs[i]) and self.Filter_bricks(childs[i + 1]):
                    two_gen['brick1'].append(childs[i])
                    two_gen['brick2'].append(childs[i + 1])
                    two_gen_lst.extend([childs[i], childs[i + 1]])
            # print(two_gen)
            # print(two_gen_lst)
            two_gen_set = set(two_gen_lst)
            two_gen_lst = list(two_gen_set)
            return two_gen_lst

        elif self.args.cut_num == 2:
            # three_gen = {'brick1': [], 'scaffold': [], 'brick2': []}
            three_gen_lst = []
            for index, child in enumerate(self.recap_res.children.values()):
                if child.children:
                    if index % 2 == 0 and self.Filter_bricks(childs[index + 1]):
                        grandchildren = list(child.children.keys())
                        for i in range(0, len(grandchildren), 2):
                            if self.Filter_bricks(grandchildren[i]) and self.Filter_bricks(grandchildren[i + 1]):
                                three_gen_lst.extend([grandchildren[i], grandchildren[i + 1], childs[index + 1]])

                    elif index % 2 != 0 and self.Filter_bricks(childs[index - 1]):
                        grandchildren = list(child.children.keys())
                        for i in range(0, len(grandchildren), 2):
                            if self.Filter_bricks(grandchildren[i]) and self.Filter_bricks(grandchildren[i + 1]):
                                three_gen_lst.extend([childs[index - 1], grandchildren[i], grandchildren[i + 1]])

            # three_gen_lst = self.remove_duplicates(three_gen_lst)

        # return self.get_leaves(self.recap_res)
        three_gen_set = set(three_gen_lst)
        three_gen_lst = list(three_gen_set)
        return three_gen_lst


    def search_brick(self, smile,scaffold_flag):
        threshold = self.args.threshold
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


    def replace_dummys(self,initial_results):
        '''
        Replace dummy atoms in the fragments with environment atoms
        :param initial_results:
        :return:
        '''
        flattened_list = set(initial_results)
        # print(flattened_list)
        replace_dict = {key: [] for key in flattened_list}
        # Replace atoms at breaking points
        search_dict = {key: [] for key in flattened_list}

        for fragment in flattened_list:
            pattern_match = re.findall(self.pattern, fragment)
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
        print(replace_dict)

        for key, values in tqdm(replace_dict.items(), desc="Processing keys in replace_dict"):
            try:
                scffold_flag = len(re.findall(self.pattern, key)) >= 2
                for value in values:
                    sim_list = self.search_brick(value, scffold_flag)
                    if len(sim_list) > 0:
                        search_dict[key].extend(sim_list)
            except Exception as e:
                print(value)
                print(e)
                continue

        # Deduplicate the values for each key in search_dict
        for key, value in tqdm(search_dict.items(), desc="Deduplicating search_dict values"):
            search_dict[key] = list(set(value))

        print(search_dict)

        dfs = {}
        for key, value in tqdm(search_dict.items(), desc="Generating dataframes"):
            dfs[key] = self.brick_dfs(value)

        return replace_dict, search_dict, dfs



def parse_args():
    parser = argparse.ArgumentParser(description='Ligand Fragmentation')
    parser.add_argument('-i', '--input', type=str, default='FC1=CC(OC)=C(C2=C(Cl)C=NC(NC3=NN(C4CCNCC4)C=C3)=C2)C=C1',
                        help='ligand SMILES')
    # parser.add_argument('-n', '--name', type=str, default="FGFR", help='compound name')
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 处理scPDB中的配体
    # extract_scpdb_ligand()
    # break_scpdb_ligand()
    # scpdb_brick_analysis()

    # amg_650 = 'Cc1cc(NC(=O)c2ccc(NS(=O)(=O)CCO)cc2N2CCC3(CC2)CC3)nc(N2CCC(F)(F)CC2)n1'
    # negishi = 'O=C(CCC(=O)N1CCN(S(=O)(=O)C2=CC=CC=C2)CC1)N1CCC(CCC2=CC=CC=C2)CC1'
    # il_17 = 'CC(C)[n]1nccc1C(N[C@@H](C(C1CC1)C1CC1)C(Nc(cc1)ccc1-c1c(C)[nH]nc1C)=O)=O'
    # test = 'O=C(c1c(N2CCC3(CC3)CC2)cc(NS(=O)(CCO)=O)cc1)O'
    # eticlopride = 'CCC1=C(C(C(NC[C@@H]2CCCN2CC)=O)=C(C(Cl)=C1)OC)O'
    # trem2 = 'CC1=NC=CC(C2CN(CCS2)C3=NC(C4=CC=C(C=C4F)Cl)=C5N=C(N(C(C5=N3)=O)C)C)=C1'
    # LT_142 = 'CC1=NC2=C(N=C(N=C2C(N1C)=O)N3CCO[C@@H](C3)C4=CN(N=C4)C)C5=CC=C(C=C5F)Cl'
    # CDK = 'FC1=CC(OC)=C(C2=C(Cl)C=NC(NC3=NN(C4CCNCC4)C=C3)=C2)C=C1'
    # AC_484 = 'CC(CCNC1CCC2=CC(O)=C(C(F)=C2C1)N3CC(NS3(=O)=O)=O)C'
    # PTPN2 = 'O=S(NC1=NC=CS1)(C2=CC=C(NS(=O)(C3=CC(OC(CC)=C4C(C5=CC(Br)=C(O)C(Br)=C5)=O)=C4C=C3)=O)C=C2)=O'
    # S484 = 'O=C=NS(=O)(Cl)=O'
    # tl1_8020_5400 = 'O=C(CC1=NC(C2=C(C=CC=C2)O)=NN1)O'
    # tl1_S823_4091 = 'CC(N[C@H]1CN(C[C@@H]1CO)C(C2=NNC3=CC=CC=C32)=O)=O'
    # tl1_LT_2017_702 = 'COC(=O)C1CCC(NC(=O)C2=CC(C3=NNC4=C3C=C(C3=NN=CN3C(C)C)C=C4)=CC=C2)CC1'
    # tl1_D278_0479 = 'NC(=O)C1=CC=C(NS(=O)(=O)CC2=CC=CC=C2)C=C1'
    # tl1_Y030_3692 = 'CC1=NC(C2=CC=CC=C2)=CC(NC2=CC(C(=O)O)=C(O)C=C2)=N1'
    # jmc = 'O=C(NC1=CC=CC(C2(C(F)(F)F)NN2)=C1)C1=CC=C(NS(=O)(=O)CCO)C=C1N1CCC2(CC1)CC2'

    args = parse_args()
    # 打碎配体
    start_time = time.time()
    ts = CompoundSplit(args)

    break_result_lst = ts.decompose()
    replace_result, search_result, dfs = ts.replace_dummys(break_result_lst)
    query_frags = []
    match_bbid = []
    for key, value in replace_result.items():
        query_frags += value
    with open('data/query_frags.txt', 'w') as f:
        for frag in query_frags:
            f.write(frag + '\n')
    for key, value in search_result.items():
        match_bbid += value
    print('query_frags :', len(query_frags))
    print('match_bbid :', len(match_bbid))
    bb_data = pd.read_feather('../../Data/2_rules/enamine_bbcat_to_rules.feather')
    # 根据match_bbs中的ID从bb_data中提取smiles
    match_bbs = set()
    for i in match_bbid:
        match_bbs.add(bb_data.loc[bb_data['ID'] == i]['SMILES'].values[0])
    match_bbs = list(match_bbs)
    print('match_bbs :', len(match_bbs))
    # rule_info = pd.read_excel('../../Data/2_rules/rules.xlsx')
    # rules = rule_info['Smirks'].tolist()

    with open('../../Data/2_rules/rxn_set_uspto.txt', 'r') as f:
        rules = [l.strip() for l in f.readlines()]

    filter_rules_path = 'data/rxn_set_filter2.txt'
    match_bb_path = 'data/match_bb.txt'
    process_reac_file(rules, match_bbs,filter_rules_path,match_bb_path)

    # ts.get_result(break_result_lst,dfs)