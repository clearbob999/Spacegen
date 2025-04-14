import argparse
import pandas as pd
import glob
import os
from rdkit import Chem
from tqdm import tqdm


class BrickAnalysisCsv():
    """
    Summarize all bricks and create a brick reaction table.
    """
    def __init__(self,brick_dir,rules_path,output_path,save_pkl):
        self.brick_dir = brick_dir
        self.reactions = os.listdir(brick_dir)
        self.df = pd.DataFrame(columns=['SMILES', 'ID'] + self.reactions)
        self.rules = pd.read_excel(rules_path)
        self.output_path = output_path
        self.save_pkl = save_pkl
        self.chem_group = chem_group

    def process_reaction_csv(self,reaction):
        """
        Input the reaction name and process the corresponding CSV file.
        :param reaction: reaction name
        """
        reactant1_brick = pd.DataFrame(columns=['SMILES', 'ID'] + self.reactions)
        reactant2_brick = pd.DataFrame(columns=['SMILES', 'ID'] + self.reactions)

        csv_path = os.path.join(self.brick_dir,reaction)
        file_paths = glob.glob(f'{csv_path}/*.csv')

        target_row = self.rules[self.rules['Name_reactions'] == reaction]
        reactant1 = target_row['Reactant1'].iloc[0]
        reactant2 = target_row['Reactant2'].iloc[0]

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            brick = file_name.split('_')[0]
            if brick in reactant1:
                reactant1_df = pd.read_csv(file_path, usecols=['SMILES', 'ID'])
                reactant1_brick = pd.concat([reactant1_brick, reactant1_df], axis=0)
                reactant1_brick.drop_duplicates(subset=['ID'], keep='first', inplace=True)
            elif brick in reactant2:
                reactant2_df = pd.read_csv(file_path, usecols=['SMILES', 'ID'])
                reactant2_brick = pd.concat([reactant2_brick, reactant2_df], axis=0)
                reactant2_brick.drop_duplicates(subset=['ID'], keep='first', inplace=True)
            else:
                print(f'{file_path} is not in reactants')

        # Merge reactant1_brick and reactant2_brick into df
        print(f'Length before adding {reaction}: {len(self.df)}')
        self.df = pd.concat([self.df, reactant1_brick, reactant2_brick], axis=0)
        print(f'Length after adding {reaction}: {len(self.df)}')

        self.df.loc[self.df['ID'].isin(reactant1_brick['ID']), reaction] = 1
        self.df.loc[self.df['ID'].isin(reactant2_brick['ID']), reaction] = 2

        reactant_df = pd.concat([reactant1_brick, reactant2_brick], axis=0)
        duplicate = reactant_df[reactant_df.duplicated(subset=['ID'], keep=False)]

        self.df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
        print(f'Length after removing duplicates for {reaction}: {len(self.df)}')

        # Set the value of the reaction column to 3 for duplicate IDs
        self.df.loc[self.df['ID'].isin(duplicate['ID']), reaction] = 3

    def brick_analysis_csv(self):
        if os.path.exists(self.output_path):
            return print(f'{self.output_path} exist.')
        for reaction in self.reactions:
            print(f'Adding {reaction} reaction')
            self.process_reaction_csv(reaction)

        # Save the result, fill remaining empty values with 0
        self.df.fillna(0, inplace=True)
        self.df.to_csv(self.output_path, index=False)
        print(f'{self.output_path} saved successfully.')
        if self.save_pkl:
            # self.df['MOL'] = self.df['SMILES'].apply(Chem.MolFromSmiles)
            self.df.to_pickle(self.output_path.replace('.csv','.pkl'))
            print(f'{self.output_path} saved successfully,mol transform.')


    def extract_scaffold(self):
        """
        Extract scaffolds containing at least two functional groups.
        """
        print("Extracting scaffolds containing two functional groups...")
        self.scaffold_output_path = self.output_path.replace('.csv', '_scaffold.csv')
        # 如果MOL列不存在，则不用转换
        if 'MOL' not in self.df.columns:
            self.df = pd.read_csv(self.output_path)
            self.df['MOL'] = self.df['SMILES'].apply(Chem.MolFromSmiles)
        results = []

        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            mol = row['MOL']
            if mol is None:
                results.append(False)
                continue

            match_counts = {group: len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
                            for group, smarts in self.chem_group.items()}
            count = sum(1 for v in match_counts.values() if v >= 1)
            multi_functional_groups = any(v > 1 for v in match_counts.values())

            if count >= 2 or multi_functional_groups:
                results.append(True)
            else:
                results.append(False)

        self.df['Contains_Two_Functional_Groups'] = results
        scaffold_df = self.df[self.df['Contains_Two_Functional_Groups']].copy()
        scaffold_df.drop(columns=['MOL','Contains_Two_Functional_Groups'], inplace=True)

        # Save scaffold data to CSV
        scaffold_df.to_csv(self.scaffold_output_path, index=False)
        print(f"Scaffold extraction saved to {self.scaffold_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='bricks to rules')
    parser.add_argument('--process_dir', type=str, default='../../Data/1_brickdata/brick_process',
                        help='The directory where reaction bricks will be saved')
    parser.add_argument('--rule_path', type=str, default='../../Data/2_rules/rules.xlsx',
                        help='The path of reaction rule file')
    parser.add_argument('--outpath', type=str, default='../../Data/2_rules/enamine_bbcat_to_rules.csv',
                        help='Path to save the number of bricks corresponding to each reaction')
    parser.add_argument('--save_pkl',default=True, help='Save the pkl file')
    parser.add_argument('--scaffold',default=False, help='Extract the scaffold')
    return parser.parse_args()

if __name__ == '__main__':
    chem_group = {'Esters': '[*]C(=O)OC[*]', 'Alcohols': '[C;$([C]-[OH]);!$([C](=O)[O])]'
        , 'Acylhalides': '[*]C(=O)[F,Cl,Br,I]', 'SulfonylHalides': '[*]-S(=O)(=O)-[F,Cl,Br,I]'
        , 'AlkylHalides': '[Cl,Br,I,F][CX4;!$(C-[!#6;!#1])]', 'Primary_Amines': '[NX3;H2;!$(N-*);!$(N~[!#6;!#1])]'
        , 'Secondary_Amines': '[NX3;H1;!$(N-*);!$(N~[!#6;!#1])]', 'CarboxylicAcids': 'C(=O)[OH1]', 'Nitriles': 'C#N'
        , 'Dienes': 'C=CC=C', 'Vinyls': '[CH2]=[CH]', 'Acetylenes': 'C#[CH]', 'Allenes': 'C=C=C', 'Boronates': 'B(O)O'
        , 'ArylHalides': '[Cl,Br,I,F][c][*]', 'Ketones': '[#6][C](=O)[#6]', 'Alkynes': 'C#C'
        , 'Sulfonyl chlorides': '[*]-S(=O)(=O)-[Cl]', 'Imides': '[NH1;$(N(C=O)C=O)]',
                  'Sulfonamide': '[NH1;$(N([#6])S(=O)=O)]'
        , 'Thiols': '[SX2;H1]', 'Aldehydes': '[CX3H1](=O)[#6]'}
    args = parse_args()
    brick_analysis = BrickAnalysisCsv(args.process_dir,args.rule_path,args.outpath,args.save_pkl)
    brick_analysis.brick_analysis_csv()
    brick_analysis.extract_scaffold()
