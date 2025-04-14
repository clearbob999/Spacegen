import argparse
from multiprocessing import Pool
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem,SDWriter, QED, Descriptors, rdMolDescriptors, inchi, RDConfig
from rdkit.Chem.rdfiltercatalog import FilterCatalog, FilterCatalogParams

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import time
import warnings
from tqdm import tqdm

# 禁用RDKit的警告
RDLogger.DisableLog('rdApp.*')

# 忽略"Omitted undefined stereo"警告
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")

class ReactionEnumerator(object):
    def __init__(self,reactant1_path, reactant2_path,reaction_name, rules, args):
        self.args = args
        self.workers = args.workers
        self.reactant1_path = reactant1_path
        self.reactant2_path = reactant2_path
        self.reaction_type = reaction_name
        self.rules = rules
        self.csv1_input, self.csv2_input = pd.read_csv(self.reactant1_path), pd.read_csv(self.reactant2_path)
        self.output_dir = os.path.join(args.output, args.format, self.reaction_type)

    def _split_list(self,sdf1_lst,n):
        '''将列表分成n份,要求子列表元素长度尽可能接近
        lst待拆分列表
        n 子列表个数
        '''
        if len(sdf1_lst) >= n:
            res = [[] for _ in range(n)]
            avg_len, rem = divmod(len(sdf1_lst), n)
            start = 0
            for i in range(n):
                length = avg_len + 1 if i < rem else avg_len
                res[i] = sdf1_lst[start:start + length]
                start += length
        else:
            res = [[i] for i in sdf1_lst]
            self.workers = len(sdf1_lst)
        return res

    def RXN(self,reactant1,reactant2,progress_index,rxn):
        """
        产物和砌块的联系待建
        键为这些'Reactant1','Reactant1_ID','Reactant2','Reactant2_ID','Products'
        """
        # 要不弄五个列表，然后合并 试下dataframe？
        # res = pd.DataFrame(columns=['Reactant1','Reactant1_ID','Reactant2','Reactant2_ID','Products'])
        res = []
        total_iterations = len(reactant1)
        progress_bar = tqdm(total=total_iterations, desc="进程%d反应枚举进度" % progress_index)
        for index1, row1 in tqdm(reactant1.iterrows(), desc="进程%d反应枚举进度" % progress_index):
            for index2,row2 in reactant2.iterrows():
                reaction_result = rxn.RunReactants((row1['Mol1'], row2['Mol2']))
                for i in range(len(reaction_result)):
                    mol = reaction_result[i][0]
                    product = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False, allHsExplicit=False)
                    new_row = {'Reactant1': row1['Reactant1'],
                               'Reactant1_ID': row1['Reactant1_ID'],
                               'Reactant2': row2['Reactant2'],
                               'Reactant2_ID': row2['Reactant2_ID'],
                               'Products': product}
                    res.append(new_row)
            progress_bar.update(1)
        progress_bar.close()  # 关闭进度条
        res_df = pd.DataFrame(res)
        return res_df

    def calculate_property(self,product_smiles,output_path,nthreads):
        '''计算分子的性质'''
        start_time = time.time()
        data = []
        print('开始计算性质')
        pool = Pool(processes=nthreads)
        for i, smiles in enumerate(product_smiles):
            pool.apply_async(self.cal_mol_props, args=(smiles, i,), callback=lambda x: data.append(x))
        pool.close()
        pool.join()
        print('性质计算完成,耗时{}s'.format(time.time() - start_time))
        merged_df = pd.concat(data, ignore_index=True)
        result_outputpath = os.path.join(self.output_dir, 'property')
        os.makedirs(result_outputpath, exist_ok=True)
        merged_df.to_csv(os.path.join(result_outputpath , os.path.basename(output_path)), index=False)
        # with open(output_path + '.pkl', 'wb') as f:
        #     pkl.dump(merged_df,f)
        # with open(output_path, 'w') as file:
        #     file.write(merged_df.to_string(index=False))
        print('性质计算保存成功,耗时{}s'.format(time.time() - start_time))

    def check_pains(self,mol):
        params_pains = FilterCatalogParams()
        params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        catalog_pains = FilterCatalog(params_pains)
        return catalog_pains.HasMatch(mol)

    def cal_mol_props_pandarallel(self,smi, verbose=False):
        try:
            m = Chem.MolFromSmiles(smi)
            if not m:
                return None, None, None, None, None, None, None, None, None, None, None, None
            mw = Descriptors.MolWt(m)
            hac = Descriptors.HeavyAtomCount(m)
            clogp = Descriptors.MolLogP(m)
            hba = Descriptors.NumHAcceptors(m)
            hbd = Descriptors.NumHDonors(m)
            rob = Descriptors.NumRotatableBonds(m)
            fsp3 = Descriptors.FractionCSP3(m)
            tpsa = Descriptors.TPSA(m)
            qed = QED.qed(m)
            SAscore = sascorer.calculateScore(m)
            InchiKey = inchi.InchiToInchiKey(inchi.MolToInchi(m))
            pains_flag = self.check_pains(m)

            if verbose:
                print('Mw ', mw)
                print('HAC ', hac)
                print('cLogp ', clogp)
                print('HBA ', hba)
                print('HBD ', hbd)
                print('RotB ', rob)
                print('TPSA ', tpsa)
                print('QED ', qed)
                print('SAscore ', SAscore)
                print('PAINS', pains_flag)
            return mw, hac, clogp, hba, hbd, rob, fsp3, tpsa, qed, SAscore, InchiKey, pains_flag
        except Exception as e:
            print(e)
            return None, None, None, None, None, None, None, None, None, None, None, None

    def calculate_property(self,output_path,nthreads):
        df = pd.read_csv(output_path)
        df.rename({'Products':'Smiles'},axis=1,inplace=True)
        pandarallel.initialize(nb_workers=nthreads, progress_bar=True)
        start_time = time.time()
        (df["Mw"], df["HAC"], df["cLogp"], df["HBA"], df["HBD"], df["RotB"], df["Fsp3"],
         df["TPSA"], df["QED"], df["SAscore"], df["InchiKey"], df['PAINS']) = (
            zip(*df['Smiles'].parallel_apply(self.cal_mol_props_pandarallel)))
        result_outputpath = os.path.join(self.output_dir, 'property')
        df.to_csv(os.path.join(result_outputpath , os.path.basename(output_path)), index=False)
        end_time = time.time()
        print('Time:', end_time - start_time)

    def run_reaction(self):
        # 定义输出文件名
        output_name = (('_').join(os.path.basename(self.reactant1_path).split('_')[:-1]) + '_'
                       + ('_').join(os.path.basename(self.reactant2_path).split('_')[:-1]))
        rxn = AllChem.ReactionFromSmarts(self.rules)

        reactant1_csv = self.csv1_input[['SMILES','ID']].copy()
        reactant1_csv.loc[:,'Mol1'] = reactant1_csv['SMILES'].apply(Chem.MolFromSmiles)
        reactant1_csv.rename(columns={'SMILES':'Reactant1','ID':'Reactant1_ID'},inplace=True)

        reactant2_csv = self.csv2_input[['SMILES', 'ID']].copy()
        reactant2_csv.loc[:,'Mol2'] = reactant2_csv['SMILES'].apply(Chem.MolFromSmiles)
        reactant2_csv.rename(columns={'SMILES': 'Reactant2', 'ID': 'Reactant2_ID'}, inplace=True)

        # 要判断下产物是否会过多，以每次输出两百万为界限
        if len(reactant1_csv) * len(reactant2_csv) > 2000000:
            interval = len(reactant1_csv) // (2000000 // len(reactant2_csv))
            reactant1_first_split = self._split_list(reactant1_csv, interval)
        else:
            reactant1_first_split = [reactant1_csv]
        with tqdm(total=len(reactant1_first_split), desc=f"{self.reaction_type}总进度") as pbar_total:
            for index, reactant1_list in enumerate(reactant1_first_split):
                products = []
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(self.output_dir, f'{index}_{output_name}.{self.args.format}')
                if args.resume:
                    if not os.path.exists(output_path):
                        if len(reactant1_list) >= self.workers:
                            reactant1_second_split = self._split_list(reactant1_list,self.workers)
                        else:
                            self.workers = len(reactant1_list)
                            reactant1_second_split = self._split_list(reactant1_list, self.workers)
                        # products.append(self.RXN(reactant1_list,reactant2_csv,index,rxn))
                        pool = Pool(processes=self.workers)
                        for i,reactant1 in enumerate(reactant1_second_split):
                            pool.apply_async(self.RXN, args=(reactant1,reactant2_csv,i,rxn), callback=lambda x: products.append(x))
                        pool.close()
                        pool.join()
                        # print(products)
                        merge_products = pd.concat(products, ignore_index=True)
                        unique_products = merge_products.drop_duplicates(subset=['Products'])
                        # dupicate_products = merge_products[merge_products.duplicated(subset=['Products'], keep=False)]
                        # 存储csv文件
                        # dupicate_products.to_csv('dupicate_products.csv',index=False)
                        if self.args.format == 'csv':
                            unique_products.to_csv(output_path, index=False)
                # 把性质计算移出去，从保存的csv中读取
                if self.args.characteristic:
                    # 使用pandarallel计算性质进行对比
                    result_outputpath = os.path.join(self.output_dir, 'property')
                    os.makedirs(result_outputpath, exist_ok=True)
                    if not os.path.exists(os.path.join(result_outputpath, os.path.basename(output_path))):
                        self.calculate_property(output_path, self.workers)
            pbar_total.update(1)

def parse_args():
    parser = argparse.ArgumentParser(description='根据反应规则枚举化学空间')
    parser.add_argument('-r','--reaction', type=str, default='../../Data/2_rules/rules.xlsx'
                        , help='化学反应规则(rxn或者smirks)')
    parser.add_argument('-i','--input', type=str, help='输入的化学砌块(一般是sdf)')
    parser.add_argument("--input_reagent_1",default='./block/Enamine_CarboxylicAcids_42387cmpds_20231002.sdf'
                        , help="指定试剂1的输入文件")
    parser.add_argument("--input_reagent_2",default='./block/Enamine_Primary_Amines_41708cmpds_20231002.sdf'
                        , help="指定试剂2的输入文件")
    parser.add_argument("--input_reagent_3", help="指定试剂3的输入文件")
    parser.add_argument('-o','--3_output', type=str, default='./3_output/test_output', help='输出目录')
    parser.add_argument('-f','--format', type=str, choices=['sdf', 'smi', 'mol','csv'], default='csv'
                        , help='文件保存格式') #输入sdf会输出sdf，输入smi仅输出smi和性质
    parser.add_argument('-n','--number', type=str, help='生成的产物数量')
    parser.add_argument('-t', '--blocknumber', type=int,default=1000, help='使用砌块数量')
    # parser.add_argument('-p','--protection', type=str, help='保护基')
    parser.add_argument('--workers', type=int, default=8, help='启用进程数量')
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细日志")
    parser.add_argument("--test", action="store_true", help="开启反应测试模式")
    # 设计一个resume参数，如果resume为真，那么就从上次的位置开始，如果为假，就从头开始
    parser.add_argument("--resume", action="store_true",default=True, help="从上次的位置开始")
    parser.add_argument("-c","--characteristic", default=True, help="性质计算")
    args = parser.parse_args()
    return args

def find_file_with_prefix(folder_path, prefix):
    """查找文件夹中以指定前缀开头的文件名，并返回第一个找到的文件名，如果没有找到则返回 None"""
    for file in os.listdir(folder_path):
        if file.startswith(prefix):
            return file
    return None

if __name__ == '__main__':
    args = parse_args()
    rules = pd.read_excel('../../Data/2_rules/rules.xlsx')
    cluster_path = '../../Data/1_brickdata/brick_cluster'
    for index, row in tqdm(rules.iterrows(), total=len(rules), desc="总进度"):
        if index != 38:
            continue
        else:
            reactants_1 = row["Reactant1"].split(",")
            reactants_2 = row["Reactant2"].split(",")
            reaction_name = row["Name_reactions"]
            rules = row["Smirks"]
            folder_path = os.path.join(cluster_path, reaction_name)
            for reactant1 in reactants_1:
                target_file1 = find_file_with_prefix(folder_path, reactant1)
                if target_file1 is not None:
                    reactant1_path = os.path.join(cluster_path, reaction_name, target_file1)
                else:
                    print(f"{reaction_name}的{reactant1}不存在")
                    continue
                for reactant2 in reactants_2:
                    target_file2 = find_file_with_prefix(folder_path, reactant2)
                    if target_file2 is not None:
                        reactant2_path = os.path.join(cluster_path, reaction_name, target_file2)
                    else:
                        print(f"{reaction_name}的{reactant2}不存在")
                        continue
                    try:
                        reaction = ReactionEnumerator(reactant1_path, reactant2_path, reaction_name,rules, args)
                        reaction.run_reaction()
                    except Exception as e:
                        print(e)
                        print(f"{reaction_name}的{reactant1}+{reactant2}反应失败")
                        continue
