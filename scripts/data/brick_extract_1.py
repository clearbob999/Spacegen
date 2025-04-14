import argparse
import glob
import pickle

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem,MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
import os
from tqdm import tqdm
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D


class BrickProcess():
    """
       A class to handle the processing of chemical bricks, including the extraction of properties,
       applying reaction rules, and generating outputs in various formats such as SDF files.

       Attributes:
       -----------
       chem_group : dict
           A dictionary of predefined functional groups used for substructure matching.

       input_path : str
           The path to the file containing the properties of the chemical bricks.

       rules_path : str
           The path to the file containing the reaction rules to be applied to the bricks.

       brick_path : str
           The path to the standardized brick file.

       output_path : str
           The path where the 3_output files (such as the processed bricks) will be saved.

       save_sdf : bool
           A flag indicating whether to save the processed bricks in SDF format. Default is True.

       save_filter : bool
           A flag indicating whether to save the bricks that are filtered out during processing. Default is False.

       mol_flag : bool
           A flag indicating whether the SMILES strings in the bricks have already been converted to molecular objects (mol).
           Default is False.

       Methods:
       --------
       This class would typically include methods for processing the bricks, applying rules,
       and saving the results, which are not detailed in this snippet.
       """
    def __init__(self,property_path,rule_path,stan_brick_path,output_path):
        self.chem_group = functional_groups
        self.input_path = property_path
        self.rules_path = rule_path
        self.brick_path = stan_brick_path
        self.output_path = output_path
        self.save_sdf = True
        self.save_filter = False
        self.mol_flag = False

    def standardize_smi(self,smiles,basicClean=True,clearCharge=True, clearFrag=True, canonTautomer=False, isomeric=False):
        """
        Standardizes a given SMILES string.

        :param smiles: The SMILES string to standardize
        :param basic_clean: Remove hydrogen atoms, metals, and standardize the molecule
        :param clear_charge: Attempt to neutralize the molecule
        :param clear_frag: Keep only the main fragment as the molecule
        :param canon_tautomer: Handle tautomerization; this step may not be perfect in all cases
        :param isomeric: Set to True to retain stereochemistry; set to False to remove stereochemistry
        :return: The standardized SMILES string
        """

        try:
            clean_mol = Chem.MolFromSmiles(smiles)
            # clean_mol = Chem.SanitizeMol(mol)
            if basicClean:
                clean_mol = rdMolStandardize.Cleanup(clean_mol)
            if clearFrag:
                clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            if clearCharge:
                uncharger = rdMolStandardize.Uncharger()
                clean_mol = uncharger.uncharge(clean_mol)
            if canonTautomer:
                te = rdMolStandardize.TautomerEnumerator()  # idem
                clean_mol = te.Canonicalize(clean_mol)
            stan_smiles = Chem.MolToSmiles(clean_mol, isomericSmiles=isomeric)
        except Exception as e:
            print(f"Error standardizing SMILES: {e}, SMILES: {smiles}")
            return None
        return stan_smiles

    def standardize_brick(self):
        """
        Standardizes the SMILES strings in the input file and saves them to the 3_output file.
        """
        if os.path.exists(self.brick_path):
            return pd.read_csv(self.brick_path)
        df = pd.read_csv(self.input_path)
        df.loc[:, 'SMILES'] = df['SMILES'].apply(lambda x: self.standardize_smi(x))
        df.to_csv(self.brick_path, index=False)
        print('Standardized brick SMILES successfully.')
        return df


    def select_bricks(self,reactant_smarts, bricks_df):
        """
        Selects matching bricks based on the reactant SMARTS pattern.

        Parameters:
        - reactant_smarts: str
            The SMARTS pattern of the reactant.
        - bricks_df: pandas.DataFrame
            DataFrame containing all the bricks; must contain a 'mol' column.

        Returns:
        - pandas.DataFrame
            DataFrame of bricks that match the reactant SMARTS pattern.
        """
        # Convert the SMARTS expression into a pattern
        patt = Chem.MolFromSmarts(reactant_smarts)

        # Check if each brick matches the given pattern
        bricks_df['sub_match'] = bricks_df['mol'].apply(lambda x: x.HasSubstructMatch(patt))

        # Return the bricks that match the SMARTS pattern
        return bricks_df[bricks_df['sub_match']]


    def sub_match(self,all_bricks, reactant1, reactant2, smirk, disallowed_functions, reaction_dir):
        '''
        :param all_bricks: All preprocessed bricks
        :param reactant1: Reactant 1
        :param reactant2: Reactant 2
        :param smirk: Reaction SMIRKS expression
        :param disallowed_functions: Disallowed functional groups
        :param reaction_dir: Directory to save reaction results

        The process consists of three steps:
        1. Classify and select reactants. For complex reactants involved in cyclization reactions,
           pre-written expressions from the chemical group are used for matching.
        2. Perform substructure matching using the SMIRKS expression, filtering reactant1 and reactant2.
        3. Perform substructure matching against disallowed functions.
        '''
        # 对reactant列表进行去重

        reactant = list(set(reactant1 + reactant2))
        reactant_dict = {'reactant1': 0, 'reactant2': 0}

        for i in reactant:
            self.mol_flag = False

            # Step 1: Classify and select reactants
            if i in enamine_class:
                selected_bricks = all_bricks[all_bricks['Class'].str.contains(i, na=False)]
            elif i in enamine_subclass:
                selected_bricks = all_bricks[all_bricks['Subclass'].str.contains(i, na=False)]
            elif i in self.chem_group:
                # Convert SMILES to mol objects for substructure matching
                chem_bricks = all_bricks.copy()
                chem_bricks.loc[:, 'mol'] = all_bricks['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
                self.mol_flag = True
                reactant_function = self.chem_group[i]
                if i == 'Vinyl_arylhalides':
                    for j in self.chem_group[i]:
                        chem_bricks = self.select_bricks(j, chem_bricks)
                else:
                    chem_bricks = self.select_bricks(reactant_function, chem_bricks)
                # Extract bricks that match the substructure
                selected_bricks = chem_bricks[chem_bricks['sub_match'] == True]
            else:
                selected_bricks = all_bricks.copy()
            print('Before substructure matching, %s brick count: %d' % (i, len(selected_bricks)))

            # Step 2: Perform substructure matching with SMIRKS
            reactant1_smarts, reactant2_smarts = smirk.split('>>')[0].split('.')
            # Convert SMILES to mol objects for substructure matching
            if not self.mol_flag:
                selected_bricks.loc[:, 'mol'] = selected_bricks['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))

            if i in reactant1:
                sign = 'reactant1'
                selected_bricks = self.select_bricks(reactant1_smarts, selected_bricks)
            else:
                sign = 'reactant2'
                selected_bricks = self.select_bricks(reactant2_smarts, selected_bricks)

            # Step 3: Perform substructure matching against disallowed functions
            if disallowed_functions != []:
                for j in disallowed_functions:
                    disallowed_function = self.chem_group[j]
                    patt = Chem.MolFromSmarts(disallowed_function)
                    selected_bricks.loc[:, 'sub_match'] = selected_bricks['mol'].apply(lambda x: x.HasSubstructMatch(patt))

                    # # Save the filtered bricks for review if required
                    filtered_bricks = selected_bricks[selected_bricks['sub_match'] == True]
                    if self.save_filter:
                        filtered_bricks.to_csv(os.path.join(reaction_dir, f"{i}_{j}_{len(filtered_bricks)}.csv"), index=False)
                        if self.save_sdf and len(filtered_bricks) != 0:
                            writer = Chem.SDWriter(os.path.join(reaction_dir, f"{i}_{j}_{len(filtered_bricks)}.sdf"))
                            for index, row in filtered_bricks.iterrows():
                                mol = row['mol']
                                mol.SetProp("ID", row['ID'])
                                mol.SetProp("Class", row['Class'])
                                mol.SetProp("Subclass", row['Subclass'])
                                writer.write(mol)
                            writer.close()
                    selected_bricks = selected_bricks[selected_bricks['sub_match'] == False]
                    print('After substructure matching with %s, %s brick count: %d' % (j, i, len(selected_bricks)))
            reactant_dict[sign] += len(selected_bricks)
            print('After substructure matching, %s brick count: %d' % (i, len(selected_bricks)))

            selected_bricks.to_csv(os.path.join(reaction_dir, f"{i}_{len(selected_bricks)}.csv"), index=False)
            if self.save_sdf and len(selected_bricks) != 0:
                writer = Chem.SDWriter(os.path.join(reaction_dir, f"{i}_{len(selected_bricks)}.sdf"))
                for index, row in selected_bricks.iterrows():
                    mol = row['mol']
                    mol.SetProp("ID", row['ID'])
                    mol.SetProp("Class", row['Class'])
                    mol.SetProp("Subclass", row['Subclass'])
                    writer.write(mol)
                writer.close()
            print('Successfully saved bricks for %s' % i)

        if reactant1 == reactant2:
            reactant_dict['reactant2'] = reactant_dict['reactant1']
        return reactant_dict


    def extract_brick(self, process_dir):
        """
        :param process_dir: Directory where reaction bricks will be saved.
        Extracts reaction-related bricks.
        """
        # Load reaction rules
        df = pd.read_excel(self.rules_path)

        for index, row in df.iterrows():
            reactant1 = row['Reactant1'].split(',') if ',' in row['Reactant1'] else [row['Reactant1']]
            reactant2 = row['Reactant2'].split(',') if ',' in row['Reactant2'] else [row['Reactant2']]
            name_reaction = row['Name_reactions']
            print(f"Processing reaction {index}: {name_reaction}")

            reaction_dir = os.path.join(process_dir, f'{name_reaction}')
            if not os.path.exists(reaction_dir):
                os.makedirs(reaction_dir, exist_ok=True)
            else:
                continue

            # Check for disallowed functions, if any
            if isinstance(row['Disallowed functions'], str):
                disallowed_functions = row['Disallowed functions'].split(',') if ',' in row[
                    'Disallowed functions'] else [row['Disallowed functions']]
            else:
                disallowed_functions = []

            smirk = row['Smirks']
            print(reactant1, reactant2, disallowed_functions, smirk)

            all_bricks = pd.read_csv(self.brick_path)

            # Perform substructure search and filter bricks
            reactant_dict = self.sub_match(all_bricks, reactant1, reactant2, smirk,
                                           disallowed_functions, reaction_dir)
            print(reactant_dict)
            # Update 'N1' and 'N2' columns with the number of matching bricks for reactant1 and reactant2
            df.loc[index, 'N1'] = reactant_dict['reactant1']
            df.loc[index, 'N2'] = reactant_dict['reactant2']
            df.loc[index,'Products'] = reactant_dict['reactant1'] * reactant_dict['reactant2']

        df = df.sort_values(by='Products', ascending=False)
        df.to_excel(self.output_path, index=False)


    def sub_match_test(self, sdf_path , save_path, smarts_pattern):
        """
        Perform substructure matching on bricks and save matching bricks into an sdf file.
        """
        bricks = Chem.SDMolSupplier(sdf_path)
        patt = Chem.MolFromSmarts(smarts_pattern)
        writer = Chem.SDWriter(save_path)

        match_count = 0
        for i, brick in enumerate(bricks):
            if brick is not None and brick.HasSubstructMatch(patt):
                writer.write(brick)
                match_count += 1

        # Output the number of matched molecules
        print(f"Number of matched molecules: {match_count}")
        writer.close()

class BrickPropertity():
    """
    This class is designed for extracting properties from molecules stored in an SDF file
    and saving them into a CSV file. Note that pandastools may not be able to read the ID column,
    so the original approach is retained.
    """
    def __init__(self,input_path,output_path,billion_bricks):
        self.sdf_path = input_path
        self.csv_path = output_path
        self.brick_flag = billion_bricks
    def SdfToPropertycsv(self):
        if os.path.exists(self.csv_path):
            print("csv file already exists")
            return
        mols = Chem.SDMolSupplier(self.sdf_path)
        if mols is None:
            print("Failed to load molecules from the SDF file")
            return

        # pandastools_df = PandasTools.LoadSDF(self.sdf_path)
        # 107537 125991 126016 126017 148795 209678 222408 248944 251572   some property names might be missing
        # property_list = list(mols[0].GetPropNames())

        property_list = mols[0].GetPropsAsDict().keys()
        # property_list = ['Mw', 'ID', 'IUPAC Name', 'URL', 'Stock_weight_G', 'Class', 'Subclass']
        data = []

        for i, mol in tqdm(enumerate(mols), total=len(mols), desc='Extracting properties'):
            try:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    # Extract other properties of the molecule
                    mol_properties = {prop: mol.GetProp(prop) if mol.HasProp(prop) else '' for prop in property_list}
                    data.append({'SMILES': smiles, **mol_properties})
            except Exception as e:
                print(f"Error processing molecule at index {i}: {e}")
        df = pd.DataFrame(data)
        if self.brick_flag:
            self.standardize_brick_name(df)
        else:
            df.to_csv(self.csv_path, index=False)
            print(f"Successfully saved properties to {self.csv_path}")
            return df

    def extract_class(self):
        '''
        Extract information from the 'Class' and 'Subclass' columns
        '''
        class_set,subclass_set = set(),set()
        df = pd.read_csv(self.csv_path)

        # Ensure that 'Class' and 'Subclass' are of string type and handle NaN values
        df['Class'] = df['Class'].astype(str).replace('nan', '')
        df['Subclass'] = df['Subclass'].astype(str).replace('nan', '')

        # Extract all values from the 'Class' and 'Subclass' columns
        for _, row in df.iterrows():
            if row['Class']:
                class_set.update(value.strip() for value in row['Class'].split(',') if value.strip())
            if row['Subclass']:
                subclass_set.update(value.strip() for value in row['Subclass'].split(',') if value.strip())

        print("Class values:", class_set)
        print("Class set length:", len(class_set))

        print("Subclass values:", subclass_set)
        print("Subclass set length:", len(subclass_set))

    def standardize_brick_name(self,df):
        '''
        Standardize the names of molecular building blocks
        '''

        df.loc[:, 'Class'] = df['Class'].apply(lambda x:x.replace('Primary Amines','Primary_Amines')
                                               .replace('Secondary Amines','Secondary_Amines')
                                               .replace('Boronic Acids and Derivatives','Boronics')
                                               .replace('Organoelement Compounds','ElementOrganics'))
        df.loc[:, 'Class'] = df['Class'].apply(
            lambda x: x.replace('1,2-', '12').replace('1,3-', '13').replace('1,4-', '14').replace(' ',''))
        df.to_csv(self.stan_csv_path, index=False)
        print('Standardization of molecular building block names successful')


class BrickAnalysisCsv():
    """
    Summarize all bricks and create a brick reaction table.
    """
    def __init__(self,brick_dir,rules_path,output_path):
        self.brick_dir = brick_dir
        self.reactions = os.listdir(brick_dir)
        self.df = pd.DataFrame(columns=['SMILES', 'ID'] + self.reactions)
        self.rules = pd.read_excel(rules_path)
        self.output_path = output_path
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
        # self.df.reset_index(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df.to_feather(self.output_path)
        print(f'{self.output_path} saved successfully.')

    def fp_generate(self,fp_save_path,fp_type):
        """
        Generate fingerprints for the bricks and save them to a pkl file.
        """
        if os.path.exists(fp_save_path):
            return
        data = pd.read_feather(self.output_path)
        data['MOL'] = data['SMILES'].apply(Chem.MolFromSmiles)

        # Generate fingerprints based on the selected type
        if fp_type == 'morgan':
            # Generate Morgan fingerprints (default is radius=2, nBits=1024)
            data['Fingerprint'] = data['MOL'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024))

        elif fp_type == 'maccs':
            # Generate MACCS keys
            data['Fingerprint'] = data['MOL'].apply(lambda x: MACCSkeys.GenMACCSKeys(x))

        elif fp_type == 'pham2D':
            # Generate Pharm2D fingerprints
            factory = Gobbi_Pharm2D.factory
            data['Fingerprint'] = data['MOL'].apply(lambda x: Generate.Gen2DFingerprint(x, factory))

        else:
            print(f"Error: Unknown fingerprint type '{fp_type}'")
            return

        fingerprints = data['Fingerprint'].tolist()
        with open(fp_save_path,'wb') as f:
            pickle.dump(fingerprints,f)

        print(f"Successfully saved fingerprints to {fp_save_path}")

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
    parser = argparse.ArgumentParser(description='Brick infomation extract and preprocess')
    parser.add_argument('--input_path', type=str, default='../../Data/1_brickdata/Enamine_Building_Blocks_1233427cmpd_20230901.sdf', help='The path of brick property sdf file')

    parser.add_argument('--fingerprint', type=str, default= 'morgan', choices=['morgan', 'maccs', 'pham2D'],
    help='The type of fingerprint. Choose from "morgan", "maccs", or "sparseIntvect".')

    parser.add_argument('--rule_path', type=str, default='../../Data/2_rules/rules.xlsx', help='The path of reaction rule file')

    parser.add_argument('--process_dir', type=str, default='../../Data/1_brickdata/brick_process', help='The directory where reaction bricks will be saved')

    parser.add_argument('--scaffold', default=False, help='Extract the scaffold')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    brick_catagory = ['enamine_stock','enamine_bbcat','mcule','Fragment']
    chem_group = {'Esters': '[*]C(=O)OC[*]', 'Alcohols': '[C;$([C]-[OH]);!$([C](=O)[O])]'
        , 'Acylhalides': '[*]C(=O)[F,Cl,Br,I]', 'SulfonylHalides': '[*]-S(=O)(=O)-[F,Cl,Br,I]'
        , 'AlkylHalides': '[Cl,Br,I,F][CX4;!$(C-[!#6;!#1])]', 'Primary_Amines': '[NX3;H2;!$(N-*);!$(N~[!#6;!#1])]'
        , 'Secondary_Amines': '[NX3;H1;!$(N-*);!$(N~[!#6;!#1])]', 'CarboxylicAcids': 'C(=O)[OH1]', 'Nitriles': 'C#N'
        , 'Dienes': 'C=CC=C', 'Vinyls': '[CH2]=[CH]', 'Acetylenes': 'C#[CH]', 'Allenes': 'C=C=C', 'Boronates': 'B(O)O'
        , 'ArylHalides': '[Cl,Br,I,F][c][*]', 'Ketones': '[#6][C](=O)[#6]', 'Alkynes': 'C#C'
        , 'Sulfonyl chlorides': '[*]-S(=O)(=O)-[Cl]', 'Imides': '[NH1;$(N(C=O)C=O)]',
                  'Sulfonamide': '[NH1;$(N([#6])S(=O)=O)]'
        , 'Thiols': '[SX2;H1]', 'Aldehydes': '[CX3H1](=O)[#6]'}

    if 'Stock' in args.input_path:
        brick_flag = brick_catagory[0]
    elif 'mcule' in args.input_path:
        brick_flag = brick_catagory[2]
    elif 'Fragment' in args.input_path:
        brick_flag = brick_catagory[3]
    else:
        brick_flag = brick_catagory[1]

    fp_flag = args.fingerprint
    property_path = f'../../Data/1_brickdata/{brick_flag}_brick_class.csv'
    stan_brick_path = f'../../Data/1_brickdata/{brick_flag}_stan_brick_class.csv'
    num_output_path = f'../../Data/2_rules/{brick_flag}_rules_new.xlsx'
    brick2rules_output_path = f'../../Data/2_rules/{brick_flag}_to_rules.feather'
    fp_output_path = f'../../Data/2_rules/{brick_flag}_{fp_flag}_fp.pkl'
    brick_process_dir = os.path.join(args.process_dir, brick_flag)

    functional_groups = {'Esters': '[*]C(=O)OC[*]', 'Alcohols': '[C;$([C]-[OH]);!$([C](=O)[O])]'
        , 'Acylhalides': '[*]C(=O)[F,Cl,Br,I]', 'SulfonylHalides': '[*]-S(=O)(=O)-[F,Cl,Br,I]'
        , 'AlkylHalides': '[Cl,Br,I,F][CX4;!$(C-[!#6;!#1])]', 'Primary_Amines': '[NX3;H2;!$(N-*);!$(N~[!#6;!#1])]'
        , 'Secondary_Amines': '[NX3;H1;!$(N-*);!$(N~[!#6;!#1])]', 'CarboxylicAcids': 'C(=O)[OH1]', 'Nitriles': 'C#N'
        , 'Dienes': 'C=CC=C', 'Vinyls': '[CH2]=[CH]', 'Acetylenes': 'C#[CH]', 'Allenes': 'C=C=C',
                         'Boronates': '[B][O][C]'
        , 'ArylHalides': '[Cl,Br,I,F][c][*]', 'Ketones': '[#6][C](=O)[#6]', 'Alkynes': 'C#C'
        , 'Sulfonyl chlorides': '[*]-S(=O)(=O)-[Cl]', 'Imides': '[NH1;$(N(C=O)C=O)]',
                         'Sulfonamide': '[NH1;$(N([#6])S(=O)=O)]'
        , 'Vinyl_arylhalides': ['[Cl,Br,I,F][c][*]', '[CH2]=[CH]', 'c1ccccc1']
        , 'Diaminoaryl': '[cH1:1]1:[c:2](-[NH2:7]):[c:3](-[NH1;D2:8]):[c:4]:[c:5]:[c:6]:1'
        , '1_hydroxy_2_aminoheteroaromatic': '[cH1:1]1:[c:2](-[NH2:7]):[c:3](-[OH1:8]):[c:4]:[c:5]:[c:6]:1'
        , 'o_aminophenol': '[cH1:1]1:[c:2](-[NH2:7]):[c:3](-[OH1:8]):[c:4]:[c:5]:[c:6]:1'
        , 'Benzaldehydes': '[cH1:1]1:[c:2](-[CH1:7]=[OD1]):[c:3]:[c:4]:[c:5]:[c:6]:1'
        , 'Benzoic_acids': '[cH1:1]1:[c:2](-[C:5](=[OD1])-[OH1]):[c:3]:[c:4]:[c:5]:[c:6]:1'
        , 'Hydroxyacetophenones': '[cH1:1]1:[c:2](-[C](=O)[CH3]):[c:3](-[OH1:8]):[c:4]:[c:5]:[c:6]:1'
                         }

    if 'Stock' in args.input_path:
        millionbrick_flag = False
        enamine_class = ['Scaffolds', 'other', 'Acylhalides', 'Primary_Amines', 'CarboxylicAcids', 'Alcohols', 'Reagents'
            , 'SulfonylHalides', 'ArylHalides', 'MedChemHighlights', 'Aminoacids', 'Bifunctional', 'Alkenes', 'Acetylenes'
            , 'Amides', '12bisNucleophiles', 'Sulfinates', '13bisElectrophiles', 'Secondary_Amines', 'Trifunctional'
            , 'Esters', '13bisNucleophiles', 'Aldehydes', '12bisElectrophiles', 'Ketones', '14bisNucleophiles', 'Boronics'
            , 'AlkylHalides', 'ElementOrganics', 'Azides']
        enamine_subclass = ['EnaminoIminiates', 'pPhSatBioisosteres', 'Amides', 'Amino_ArX_Nitro',
                            'Aryl_halides_fluorinated'
            , 'SCN_acid_', 'Heteroaromatic_benzyl_halides', 'HetPropionic_Acids', 'Nacylhalides', 'Ketones'
            , 'aliphatic_boronates', 'Oximes', 'AlkenesCH2', 'Fluorinated_Aldehydes', 'Fluorinated_Acids'
            , 'NaturalAmino_Acids', 'r7_r8_aliphatic_amines', 'Acid_Aldehyde_ArX', 'Esters', 'Fluorinated_NAliphatic_AA'
            , 'Aliphatic_Aldehydes', 'FAlkEthers', 'UnusualFAlk', 'Halocarbonates', 'Cyc-Anilines', 'Azides_Het-benzylic'
            , 'bKetoEsters', 'HydrazidesNHNH2', '13diCarbonyls', 'Azides_CH2Ar', 'HetAcetic_Acids', 'Hydrazones'
            , 'Acid_Ester_Nitro', 'Ene_boronates', 'EnIn_Sulfonylhalides', 'Amidines', 'AlkylSulfinates', 'SNAP'
            , 'FluoroSulfates', 'Hydroxylamines', 'PriAmines_Anilines', 'SecAmines_Aliphatic_Polycyclic', 'secHet-Anilines'
            , 'AmidesNH2', 'PriAmines_Benzylic', 'GemF2Amines', 'Ine_boronates', 'Sultames', 'Amino_Azide', 'PROTAC_talid'
            , 'Aldehyde_ArX', 'Minisci_CHpartners', 'Diazirines', 'NAcylatedAA', 'Aliphatic_aminoacid'
            , 'SecAmines_Aliphatic_Fluorinated', 'BenzBoroxoles', 'CF3py_Analogs', 'diArylAmines', 'ArylSulfinates'
            , 'WeinrebAmides', 'Isonitriles', 'NfmocAA_Nitro', 'NbocAA_Nitro', 'Aldehyde_Ester', 'Aliphatic_Sulfonylhalides'
            , 'PriAmines_scaffold', 'Benzyl_halides', 'betaAA', 'NfmocAA', 'Thioles', 'Ester_Isocyanates', 'Hydrazines'
            , 'aromatic_boronates', 'Alkyne_linkers', 'NbocAA_AlkyneCH', 'SF5der', 'Acid_Ester', 'PriAmines_Het-Anilines'
            , 'Azides_cycAliphatic', 'ArPropionic_Acids', 'OxalateMonoAlkEsters', 'SecAmines_scaffold', 'NAcylatedDiamines'
            , 'Sulfoximines', 'SulfonamidesNH2', 'Azides_Het', 'secAnilines', 'Aryl_halides_Pd', 'ImidoHydroxamates'
            , 'Oxiranes', 'ThioUreasNHNH2', 'amino_boc_ester', 'Cubanes', 'ArX_CHAlkynes', 'SO2FforCoupling'
            , 'Polycyclic_Acids', 'InOnes', 'Cyc-Aliphatic', 'Acid_CHalkyne', 'other', 'tBu_Bioisosteres'
            , 'ProlineAnalogs', 'Aliphatic_Acids', 'aliphatic_alcohols', 'Azide_linkers', 'ArX_Ketones'
            , 'diSubBicyclo211hexanes', '2OBicyclo211hexanes', 'SiContBB', 'Azides_Acylated', 'ThioAmidesNH2'
            , 'secHet-benzylic', 'HeteroAliphatic_Sulfonylhalides', 'SulfoximinesMe', 'aryl_iodides', 'PriAmines_Aliphatic'
            , 'SpiroAzetidines', '2CyanoEnamines', 'Heteroaromatic_Acids', 'Acid_ArX_Nitro', 'mPhSatBioisosteres'
            , 'NfmocAA_ArX', 'DeuteratedBB', 'aHaloKetones', 'Diamines', 'JuliaKocienski', 'Aldehyde_Nboc', 'Acid_ArX_Ester'
            , 'Halooximes', 'Mg_organics', 'UreasNHNH2', 'alphaAA', 'SugarLike', 'AmidinesNHNH2', 'acid_boc_Z_'
            , 'EnylSnAlk3', 'NfmocAA_AlkyneCH', 'ArX_Azide', 'Heterocyclic_alkyl_halides', 'Acid_ArX', 'AliphBpin'
            , 'Acid_Aldehyde_AlkyneCH', 'Alkyl_halides', 'HWE', 'Bicyclo111pentanes', 'NfmocAA_Ester', 'diSpiro_func'
            , 'Acid_Aldehyde', 'HetScaff_new', 'Aromatic_aldehydes', 'Aromatic_Acids', 'AzidesAr', 'AliphBF3K', 'BF3andMIDA'
            , 'Aryl_halides', 'secBenzylic', 'IminoHydrazides', 'Azoles_NCF3', 'gammaAA', 'Nitriles', 'NOPhtal_AlkEsters'
            , 'Hydrazides', 'Aryl_halides_SN', 'Halocarbamates', 'OrtoEsters', 'Cyc-Het-benzylic', '4SubPiperidineMimics'
            , 'SO2F_AliphFunc', 'PriAmines_Aliphatic_Policyclic', 'NAr_AA', 'PriAmines_Het-benzylic', 'Cyc-Het-Anilines'
            , 'Alkenes', 'Acetylenes', 'secAliphatic', 'Heteroaromatic_Sulfonylhalides', 'Wittig', 'ArAcetic_Acids'
            , 'NbocDiamines', 'Diamines_NotherCarbamates', 'Fbenzodioxoles', 'Azides_Aliphatic', 'NcbzDiamines'
            , 'Acid_Azide', 'Amino_Ester', 'NfmocDiamines', 'NbocAA_ArX', 'nHAzoles', 'Cyc-Benzylic', 'Amine_ArX'
            , 'Tetrazines', 'NcbzAA', 'ArSiR3', 'Isothiocyanates', 'Ester_SO2X', 'Azides_CH2CAr'
            , 'Azides_Aliphatic_Fluorinated', 'PEG_linkers', 'Benzene_Sulfonylhalides', 'Hetero_aromatic_aldehydes'
            , 'Aldehydes_SNhetArylX', 'Stannates', 'Oacylhalides', 'Heterols', 'Azide_SO2X', 'SulfonylFluorides'
            , 'PhosphineOxidesR3', 'Aldehyde_Nitro', 'HydrazinesNHNH2', 'NbocAA_Ester', 'Acid_Aldehyde_Nitro'
            , 'bCarbonylNitriles', 'Acid_Nitro', 'PiperazineMimics', 'NbocNfmocAA', 'MorphMimics', 'oPhSatBioisosteres'
            , 'Fluorinated_alkyl_halides', 'PriAmines_Aliphatic_Fluorinated', 'ArSnAlk3', 'Aldehyde_SO2X', 'Oxetanes'
            , 'Michael_Acceptors', 'Acylhalides', 'Acids_scaffold', 'EnylSiR3', 'Functional_Boronates'
            , 'Polycyclic_aldehydes', 'NNitroso', 'NHPIEsters', 'Phenols', 'EnaminoKetones', 'cycSD4O', 'NbocAA'
            , 'Isocyanates', 'SNhetArylX_Boronates', 'NbnDiamines', 'OVinylKetones', 'Acetylenes_CH', 'Sulfamoylhalides']
    elif 'mcule' in args.input_path:
        millionbrick_flag = False
        enamine_class = []
        enamine_subclass = []
    else:
        millionbrick_flag = True
        enamine_class = ['TrifunctionalScaffolds', 'Sulfinates', '12bis-Nucleophiles', '12bis-Electrophiles'
            , 'Boronics', 'Ketones', 'Alkenes', 'Aldehydes', 'Amides', 'Esters', 'Azides', '13bis-Nucleophiles'
            , '14bis-Nucleophiles', 'BifunctionalScaffolds', 'Secondary_Amines', 'ArylHalides', 'AlkylHalides'
            , 'Other', 'Primary_Amines', 'MedChemHighlights', 'ElementOrganics', 'AcylHalides', 'Scaffolds'
            , '13bis-Electrophiles', 'LabReagents', 'Aminoacids', 'SulfonylHalides', 'Alcohols', 'Acetylenes'
            , 'CarboxylicAcids']
        enamine_subclass = ['', 'Cyclic Sulfates and Sulfamidates', 'Terminal Alkyne & N-Boc-Amino Acid', '<alpha>'
            , 'Amine & Aryl Halide', '<alpha>-Fused Aromatic and nitrogen-containing aliphatic compounds'
            , 'Primary Hydrazides', 'Fluorosulfates', 'Sugar-like Building Blocks', 'Amine & N-Fmoc-Amine'
            , 'Aromatic Aldehydes', 'Nitriles', 'Arylsilanes', 'Heterocyclic Sulfonyl Fluorides for Pd-Catalyzed'
            , 'Mono-N-acyl diamines', 'Diarylamines', 'Polycyclic Carboxylic Acids', 'Carboxylic acid & Nitro'
            , 'Trialkyl vinylstannanes', 'Aliphatic Amino Acids', '<beta>-Amino acrylonitriles', 'Cyclic Sulfonamides'
            , 'Alkylsulfinates', 'Epoxides', 'Ester & N-Boc-Amino Acid', '13Dicarbonyl Compounds'
            , 'Heterocyclic Alkyl Halides', 'Aliphatic cyclic compounds containing benzene ring', 'N-Hydroxyimidamides'
            , 'Aryl halides amenable to aromatic nucleophilic substitution', 'Ortoesters', 'Weinreb Amides', 'Ketones'
            , 'Aliphatic secondary amines', 'Sulfoximines', 'α-Amino Acids', '<gamma>-Arylpropionic acids'
            , 'Aldehyde & Carboxylic acid', 'Halooximes', 'Aldehyde & Sulfonyl Halide', 'Fluoroalkyl Ethers'
            , 'Acetylenes', 'Alkenes', 'Secondary N-hetarylamines', 'Arylsulphinates', 'Phenyl sulfonyl halides'
            , 'Saturated Bioisosteres of ortho- meta-Substituted Benzenes', 'Proline Analogs', 'γ-Amino Acids'
            , 'Acyl Halides', 'α-Haloketones', 'Cyclic aliphatic compounds', 'Secondary Amines Aliphatic Fluorinated'
            , '<beta>-Alkoxy <alpha>', 'Aliphatic Aldehydes', 'Alkylsulfonyl Halides', 'Diazirines', 'Aryl azides'
            , 'Aliphatic cyclic compounds containing hetroaromatic rings', 'Aminothiamides', 'Arylacetic Acids'
            , '<beta>-(Het)aryl ethylazides', 'Aliphatic N-hydroxyphthalimide esters', 'Imidohydrazides'
            , 'Benzoxaboroles', 'N-Acyl amino acids', 'Ester & Sulfonyl Halide', 'Hydrazides', 'Aryl Iodides'
            , 'Isonitriles', 'Organotrifluoroborates and MIDA Boronates', 'N-Fmoc-Amino Acid', 'Ester & Isocyanate'
            , 'Primary N-<alpha>-alkylhetaryl amines', 'Primary Thioureas', 'Carboxylic acid & Aldehyde & Aryl halide'
            , 'Carboxylic acid & Aldehyde & Nitro', 'Isothiocyanate carboxylic acids', 'Amine & N-Benzyl Amine'
            , 'N-Boc-N`-Fmoc-Diamino Acid', 'Sulfonamides', 'N-Boc-Amino Acid', 'Aliphatic Alcohols'
            , 'Unique 3D shaped Spirocycles', 'Acylated Azides', 'P(O)Me2-containing Building Blocks'
            , '2-Oxabicyclo[2.1.1]hexanes', 'Other', '<beta>-Ketoesters', 'Amine & N-Boc-Amine'
            , 'Hetaryl sulfonylhalides', 'Organostannanes', 'N-Nitrosamines', 'Thioles', 'Heterocyclic Alkohols'
            , 'Aliphatic Azides', 'Azidomethyl (hetero)aromatic compounds', 'Primary Hydrazines'
            , 'Orthogonally protected N-Boc and N-Cbz diamino acids', 'Isothiocyanates', 'Ynones'
            , 'Aromatic Heterocyclic Carboxylic Acids', 'Amidines', '<beta>-Hetarylpropionic acids'
            , 'Aliphatic Carboxylic Acids', 'Terminal Acetylenes', 'Morpholine Bioisosteres'
            , '<alpha>-Alkylhetaryl halides', 'PROTAC PEG linkers', 'Halocarbonates'
            , 'Terminal Alkyne & N-Fmoc-Amino Acid', 'Carboxylic acid & Azide', 'Silicon Containing Building Blocks'
            , 'mono-N-Boc Diamino esters', 'Primary N-arylamines', 'Halo(het)aryl ketones', 'Wittig Reagents'
            , 'SF5-containing Building Blocks', 'Functional Boronates', 'Grignard Reagents'
            , 'Functionalized Vinyl Boronates for C-C Couplings', 'Carboxylic acid & Terminal Alkyne'
            , 'Enaminoketones', 'Hydroxylamines', 'Azides Aliphatic Fluorinated', 'Analogues of CF3-Pyridine'
            , 'Arylstannanes', '<alpha>-Hetaryl alkylazides', 'Alkyl Halides', 'Alkyltrifluoroborates'
            , 'Aryl Halide & N-Fmoc-Amino Acid', 'Azide Linkers', 'Aldehyde & N-Boc-Amine'
            , 'Aliphatic Primary Amines Policyclic', 'Azide & Aryl halide', 'Nitro & N-Fmoc-Amino Acid'
            , 'Cycloalkyl azides', 'Alkyne-containing Linkers', 'Halocarbamates'
            , 'Carboxylic acid & Aldehyde & Terminal Alkyne', 'Monoalkyl oxalates', 'Fluorinated Aryl Halides'
            , 'Primary N-<alpha>-alkylaryl amines', 'Diamines', 'Carboxylic acid & Aryl halide', 'Aliphatic Amino Acid'
            , 'N`-Cbz-Amino Acid', 'Ester & N-Fmoc-Amino Acid', 'Cubanes', 'Oxetanes', 'Aromatic Boronates'
            , 'Secondary Amine-containing Scaffolds', 'Fluorinated Alkyl Halides', 'Phenols'
            , 'Protein Degrader Toolkit', 'Primary Amides', 'Carboxylic acid & Ester & Nitro'
            , 'Aryl Halide & N-Boc-Amino Acid', 'Aryl Halides for Pd-catalyzed Couplings', 'Fluorine-containing amines'
            , 'Imino enamines', 'Secondary Amines Aliphatic Policyclic', 'Aliphatic primary amines'
            , 'Polycyclic Aldehydes', 'Aldehyde & Ester', 'Heterocyclic Aldehydes', 'Hetaryl azides'
            , 'Azidosulfonyl Halides', 'Chloroformates', 'Saturated Bioisosteres of para-Substituted Benzenes'
            , 'Hydrazines', 'Fluorinated Carboxylic Acids', 'Primary N-heterylamines', 'Piperazine Bioisosteres'
            , 'Tetrazines', 'Carboxylic acid & Aryl halide & Nitro', 'Sulfamoyl Halides', 'Aromatic Acids'
            , 'Secondary N-<alpha>-alkylaryl amines', 'Aminoesters', 'Secondary N-<alpha>-alkylhetaryl amines'
            , 'Amine & N-Cbz-Amine', 'Aminoazides', 'Terminal Alkenes', 'N-(hetero)aryl amino acids', 'Alkylboronates'
            , 'Primary Amidines', '<alpha>-Aryl alkylhalides', 'Hetarylacetic Acids', '<beta>-Carbonyl nitriles'
            , 'Fluorinated Aldehydes', 'Primary Ureas', 'Aryl Halides', 'Julia–Kocienski Reagents', 'Carboxylic Acid'
            , 'Amides', 'Oximes', '<alpha>-Fused Heteroaromatic and nitrogen-containing aliphatic compounds'
            , '<beta>-unsaturated ketones', 'Heterocyclic Scaffolds', 'Trialkyl vinylsilanes', 'Carboxylic acid & Ester'
            , 'β-Amino Acids', 'Hydrazones', 'Zn-Organics', 'Aldehyde & Aryl Halide'
            , 'Saturated Bioisosteres of ortho-/meta-Substituted Benzenes', 'Aldehyde & Nitro', 'NH-Azoles'
            , 'Isocyanates', 'Primary Amines-based Scaffolds', 'Natural Amino Acids', 'Sulfonyl Fluorides'
            , 'Aryl halide - Terminal Alkyne', 'Secondary Anilines'
            , '<beta>-Unsaturated carbonyl compounds amenable to Michael addition', 'Nitro & N-Boc-Amino Acid'
            , 'Amine & Aryl Halide & Nitro', 'Aliphatic Primary Amines Fluorinated', 'Esters'
            , 'Miscellaneous amino carbamates', 'Alkynylboronic acids', 'gem-Difluorinated Amines'
            , 'Carboxylic acid & Aryl halide & Ester']

    brick_extract_propertity = BrickPropertity(args.input_path, property_path, millionbrick_flag)
    brick_property = brick_extract_propertity.SdfToPropertycsv()

    brick_processor = BrickProcess(property_path, args.rule_path, stan_brick_path, num_output_path)
    stan_df = brick_processor.standardize_brick()

    if 'mucle' not in args.input_path:
        brick_extract_propertity.extract_class()
    '''之前因为mcule没有分类信息，所以没法提取砌块，现在在提取程序里如果输入的是mcule那就不执行分类划分'''
    brick_processor.extract_brick(process_dir=brick_process_dir)

    brick_analysis = BrickAnalysisCsv(brick_process_dir, args.rule_path, brick2rules_output_path)
    brick_analysis.brick_analysis_csv()
    brick_analysis.fp_generate(fp_output_path,args.fingerprint)

    # 提取支架用
    # brick_analysis.extract_scaffold()

    # # Test substructure matching
    # smarts_pattern = '[cH1:1]1:[c:2](-[C:5](=[OD1])-[OH1]):[c:3]:[c:4]:[c:5]:[c:6]:1'
    # brick_processor.sub_match_test(args.inputpath, '../../Data/1_brickdata/sub_match_test.sdf',smarts_pattern)
