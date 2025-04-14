import argparse
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, QED, Descriptors, inchi, RDConfig
from rdkit.Chem.rdfiltercatalog import FilterCatalog, FilterCatalogParams
from pandarallel import pandarallel
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from rdkit.Contrib.SA_Score import sascorer
import time
import warnings
from tqdm import tqdm

# 禁用RDKit的警告
RDLogger.DisableLog('rdApp.*')

# 忽略"Omitted undefined stereo"警告F
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")

class ReactionEnumerator(object):
    def __init__(self,reactant1_path, reactant2_path,reaction_name, rules, args):
        self.args = args
        self.workers = args.workers
        self.reactant1_path = reactant1_path
        self.reactant2_path = reactant2_path
        self.reaction_type = reaction_name
        self.rules = rules
        self.sdf1_input, self.sdf2_input = list(Chem.SDMolSupplier(reactant1_path)), list(
            Chem.SDMolSupplier(reactant2_path))
        self.output_dir = os.path.join(args.output, self.reaction_type)

    def _split_list(self,sdf1_lst,n):
        '''
        Split a list into n parts, aiming for sublists of approximately equal length.
        Args:
            sdf1_lst: List to be split.
            n: Number of sublists.

        Returns:
            A list of n sublists.
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

    def RXN(self,sdf1,sdf2,progress_index,rxn):
        """
        Generate products from reactants.

        Args:
            sdf1: List of reactant 1 structures.
            sdf2: List of reactant 2 structures.
            progress_index: Index for progress tracking.
            rxn: Reaction object.

        Returns:
            A list of product SMILES strings.
        """
        res = []
        for reac_sdf1 in tqdm(sdf1, desc=f"Process {progress_index} Reaction Enumeration Progress"):
            for reac_sdf2 in sdf2:
                reaction_result = rxn.RunReactants((reac_sdf1, reac_sdf2))
                for i in range(len(reaction_result)):
                    mol = reaction_result[i][0]
                    product = Chem.MolToSmiles(mol, kekuleSmiles=True,isomericSmiles=False, allHsExplicit=False)
                    res.append(product)
        return res

    def duplicate(self,merge_products):
        """
        Remove duplicates from the list of products.

        Args:
            merge_products: List of product SMILES strings.

        Returns:
            A list of unique product SMILES strings.
        """
        unique_products = set()
        for product in merge_products:
            if product not in unique_products:
                unique_products.add(product)
        return list(unique_products)

    def write_smi(self,products,output_path):
        '''
        Save products as a SMILES file.

        Args:
            products: List of product SMILES strings.
            output_path: Path to the output file.
        '''
        with open(f"{output_path}", "w") as f:
            for product in products:
                f.write(product)
                f.write("\n")

    def write_sdf(self,products,output_path):
        '''
        Save products as an SDF file.

        Args:
            products: List of product SMILES strings.
            output_path: Path to the output file.
        '''
        sdf_file = Chem.SDWriter(output_path)
        num_smiles = len(products)
        with tqdm(total=num_smiles, unit="smiles") as pbar:
            if len(products) == 0:
                print('No products')
            for smile in products:
                product_mol = Chem.MolFromSmiles(smile)
                if product_mol:
                    sdf_file.write(product_mol)
            pbar.update(len(products))
        sdf_file.close()
        print('Saving successful')

    def write_csv(self,products,output_path):
        '''
        Save products as a CSV file.

        Args:
            products: List of product SMILES strings.
            output_path: Path to the output file.
        '''
        df = pd.DataFrame(products,columns=['Smiles'])
        df.to_csv(output_path,index=False)
        print('Product CSV file saved successfully')


    def calculate_property(self,product_smiles,output_path,nthreads):
        '''Calculate molecular properties.

        Args:
            product_smiles: List of product SMILES strings.
            output_path: Path to save the property data.
            nthreads: Number of threads for parallel processing.
        '''
        start_time = time.time()
        data = []
        print('Starting property calculations')
        pool = Pool(processes=nthreads)
        for i, smiles in enumerate(product_smiles):
            pool.apply_async(self.cal_mol_props, args=(smiles, i,), callback=lambda x: data.append(x))
        pool.close()
        pool.join()
        print('Property calculation completed, time elapsed {}s'.format(time.time() - start_time))
        merged_df = pd.concat(data, ignore_index=True)
        result_outputpath = os.path.join(self.output_dir, 'property')
        os.makedirs(result_outputpath, exist_ok=True)
        merged_df.to_csv(os.path.join(result_outputpath , os.path.basename(output_path)), index=False)
        print('Property calculation saved successfully, time elapsed {}s'.format(time.time() - start_time))

    def check_pains(self,mol):
        params_pains = FilterCatalogParams()
        params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        catalog_pains = FilterCatalog(params_pains)
        return catalog_pains.HasMatch(mol)

    def cal_mol_props(self, smiles, progress_index):
        """
        Calculate molecular properties: MW, HAC, slogP, HBA, HBD, RotBonds, FSP3, TPSA, QED, SAscore, InChiKey, PAINS, BRENK, NIH
        :param smiles: List of SMILES strings for the molecules
        :param progress_index: Index for progress tracking
        :return: DataFrame with calculated properties
        Note: BRENK and NIH are not currently calculated as BRENK takes over 800 seconds with 12 processes.
        """
        df = pd.DataFrame(smiles, columns=['Smiles'])
        start_time = time.time()
        df['molecule'] = df['Smiles'].apply(lambda x:Chem.MolFromSmiles(x))
        df['MW'] = df['molecule'].apply(lambda x:Descriptors.MolWt(x))
        df['HAC'] = df['molecule'].apply(lambda x:Descriptors.HeavyAtomCount(x))
        df['slogP'] = df['molecule'].apply(lambda x:Descriptors.MolLogP(x))
        df['HBA'] = df['molecule'].apply(lambda x:Descriptors.NumHAcceptors(x))
        df['HBD'] = df['molecule'].apply(lambda x:Descriptors.NumHDonors(x))
        df['RotBonds'] = df['molecule'].apply(lambda x:Descriptors.NumRotatableBonds(x))
        df['FSP3'] = df['molecule'].apply(lambda x:Descriptors.FractionCSP3(x))
        df['TPSA'] = df['molecule'].apply(lambda x:Descriptors.TPSA(x))
        df['QED'] = df['molecule'].apply(lambda x:QED.qed(x))
        df['SAscore'] = df['molecule'].apply(lambda x:sascorer.calculateScore(x))
        df['InChiKey'] = df['molecule'].apply(lambda x:inchi.InchiToInchiKey(inchi.MolToInchi(x)))
        df['PAINS'] = df['molecule'].apply(self.check_pains)
        print('Total time:', time.time() - start_time)


        df.drop(['molecule'], axis=1, inplace=True)
        return df

    def run_reaction(self):
        output_name = (('_').join(os.path.basename(self.reactant1_path).split('_')[:-1]) + '_'
                       + ('_').join(os.path.basename(self.reactant2_path).split('_')[:-1]))

        rxn = AllChem.ReactionFromSmarts(self.rules)

        # Check if the number of products might be too high; limit output to 2 million at a time
        if len(self.sdf1_input) * len(self.sdf2_input) > 2000000:
            interval = len(self.sdf1_input) // (2000000 // len(self.sdf2_input))
            sdf1_first_split = self._split_list(self.sdf1_input, interval)
        else:
            sdf1_first_split = [self.sdf1_input]
        with tqdm(total=len(sdf1_first_split), desc=f"{self.reaction_type} Total Progress") as pbar_total:
            for index, sdf1_list in enumerate(sdf1_first_split):
                products = []
                os.makedirs(self.output_dir, exist_ok=True)
                if args.resume:
                    output_path = os.path.join(self.output_dir, f'{index}_{output_name}.{self.args.format}')
                    if os.path.exists(output_path):
                        pbar_total.update(1)
                        continue
                sdf1_second_split = self._split_list(sdf1_list,self.workers)
                output_path = os.path.join(self.output_dir, f'{index}_{output_name}.{self.args.format}')
                pool = Pool(processes=self.workers)
                for i,sdf1 in enumerate(sdf1_second_split):
                    pool.apply_async(self.RXN, args=(sdf1,self.sdf2_input,i,rxn), callback=lambda x: products.append(x))
                pool.close()
                pool.join()

                merge_products = [item for sublist in products for item in sublist]
                unique_products = self.duplicate(merge_products)
                # os.makedirs(self.output_dir, exist_ok=True)

                if self.args.format == 'smi':
                    self.write_smi(unique_products,output_path)
                elif self.args.format == "sdf":
                    self.write_sdf(unique_products,output_path)
                elif self.args.format == "csv":
                    self.write_csv(unique_products,output_path)

                if self.args.characteristic:
                    unique_products_list = self._split_list(unique_products, args.workers)
                    self.calculate_property(unique_products_list, output_path, args.workers)
            pbar_total.update(1)

def find_file_with_prefix(folder_path, prefix):
    """
    Find a file in the folder that starts with the specified prefix and return the full path of the first found file. Return None if no file is found.
    :param folder_path:
    :param prefix:
    :return:
    """
    for file in os.listdir(folder_path):
        if file.startswith(prefix):
            return file
    return None

def parse_args():
    parser = argparse.ArgumentParser(description='reaction')
    parser.add_argument('--rule_path', type=str, default='../../Data/2_rules/rules.xlsx',
                        help='The path to the reaction rule file')
    parser.add_argument('--cluster_dir', type=str, default='../../Data/1_brickdata/brick_cluster',
                        help='The directory where reaction bricks will be stored')

    parser.add_argument('-i', '--input', type=str, help='Input chemical building blocks (usually in sdf format)')
    parser.add_argument("--input_reagent_1", default='../../Data/1_brickdata/CarboxylicAcids_1000.sdf',
                        help="Specify the input file for reagent 1")
    parser.add_argument("--input_reagent_2", default='../../Data/1_brickdata/Primary_Amines_1000.sdf',
                        help="Specify the input file for reagent 2")
    parser.add_argument("--input_reagent_3", help="Specify the input file for reagent 3")
    parser.add_argument('-o', '--output', type=str, default='../../Data/3_output/chem_space', help='Output directory')
    parser.add_argument('-f', '--format', type=str, choices=['smi', 'sdf', 'csv'], default='csv',
                        help='File output format')  # If input is sdf, the output will be sdf. If input is smi, it will only output smi and properties.

    parser.add_argument('--workers', type=int, default=12, help='Number of processes to use')
    # Add a resume parameter: if resume is True, start from the last position; if False, start from the beginning
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from the last position")
    parser.add_argument("-c", "--characteristic", default=True, help="Property calculation")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    rules = pd.read_excel(args.rule_path)
    cluster_dir = args.cluster_dir
    for index, row in tqdm(rules.iterrows(), total=len(rules), desc="总进度"):
        reactants_1 = row["Reactant1"].split(",")
        reactants_2 = row["Reactant2"].split(",")
        reaction_name = row["Name_reactions"]
        rules = row["Smirks"]
        folder_path = os.path.join(cluster_dir, reaction_name)
        for reactant1 in reactants_1:
            target_file1 = find_file_with_prefix(folder_path, reactant1)
            if target_file1 is not None:
                reactant1_path = os.path.join(cluster_dir, reaction_name, target_file1)
            else:
                print(f"{reaction_name}'s {reactant1} does not exist")
                continue
            for reactant2 in reactants_2:
                target_file2 = find_file_with_prefix(folder_path, reactant2)
                if target_file2 is not None:
                    reactant2_path = os.path.join(cluster_dir, reaction_name, target_file2)
                else:
                    print(f"{reaction_name}'s {reactant2} does not exist")
                    continue
                try:
                    reaction = ReactionEnumerator(reactant1_path, reactant2_path, reaction_name,rules, args)
                    reaction.run_reaction()
                except Exception as e:
                    print(e)
                    print(f"Reaction {reaction_name} with {reactant1} + {reactant2} failed")
                    continue