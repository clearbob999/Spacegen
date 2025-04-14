import argparse
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit.SimDivFilters import MaxMinPicker
import pandas as pd
from tqdm import tqdm
import os
import shutil


class MoleculePicker:
    def __init__(self, reaction_dir,cluster_dir, num_picks,draw_images
                 , radius=2, nbits=1024, seed=23):
        self.reaction_dir = reaction_dir
        self.cluster_dir = cluster_dir
        self.num_picks = num_picks
        self.radius = radius
        self.nbits = nbits
        self.seed = seed
        self.draw_images = draw_images

    def _read_sdf(self, input_file):
        molecules = []
        with Chem.SDMolSupplier(input_file) as suppl:
            for mol in tqdm(suppl, desc="Reading molecules"):
                if mol is not None:
                    molecules.append(mol)
        return molecules

    def _write_sdf(self, molecules, output_file):
        with Chem.SDWriter(output_file) as writer:
            for i in tqdm(molecules, desc="Writing to file"):
                if i is not None:
                    writer.write(i)
        print(f'{output_file} saved successfully.')

    def _draw_molecules(self, molecules, input_file, output_dir):
        for i, mol_block in enumerate(tqdm(molecules, desc="Generating molecule images")):
            os.makedirs(os.path.join(output_dir, 'pic'), exist_ok=True)
            png_name = os.path.join(output_dir, 'pic', f'{os.path.basename(input_file).split("_")[0]}_{i}.png')
            Draw.MolToFile(mol_block, png_name, dpi=1200)

    def _calculate_fps(self, molecules):
        self.fps = [AllChem.GetMorganFingerprintAsBitVect(x, self.radius, self.nbits) for x in molecules]

    def _dist_func(self, i, j):
        return 1 - DataStructs.TanimotoSimilarity(self.fps[i], self.fps[j])

    def pick_molecules(self, input_file, output_dir):
        if input_file.endswith('.sdf'):
            molecules = self._read_sdf(input_file)
        else:
            molecules = pd.read_csv(input_file)

        if len(molecules) > self.num_picks:
            self._calculate_fps(molecules)
            picker = MaxMinPicker()
            picked_indices = picker.LazyPick(self._dist_func, len(molecules), self.num_picks, seed=self.seed)
            picked_molecules = [molecules[i] for i in picked_indices]
            names = os.path.basename(input_file).split('_')[:-1]
            output_file = '_'.join(names) + f'_{self.num_picks}.sdf'

            if self.draw_images:
                self._draw_molecules(picked_molecules, input_file, output_dir)  # Draw images
            self._write_sdf(picked_molecules, os.path.join(output_dir, output_file))
            print(f'Number of selected molecules: {len(picked_molecules)}')
        else:
            output_file = os.path.join(output_dir, os.path.basename(input_file))
            shutil.copyfile(input_file, output_file)
            if self.draw_image:
                self._draw_molecules(molecules, input_file, output_dir)

    def process_reactions(self):
        for dir in os.listdir(self.reaction_dir):
            brick_path = os.path.join(self.reaction_dir, dir)
            output_dir = os.path.join(self.cluster_dir, dir)
            os.makedirs(output_dir, exist_ok=True)
            for brick_file in os.listdir(brick_path):
                if brick_file.endswith('.sdf'):
                    input_file = os.path.join(brick_path, brick_file)
                    self.pick_molecules(input_file,output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='bricks cluster')
    parser.add_argument('--process_dir', type=str, default='../../Data/1_brickdata/brick_process',
                        help='The directory where reaction bricks will be saved')
    parser.add_argument('--cluster_dir', type=str, default='../../Data/1_brickdata/brick_cluster',
                        help='The directory where reaction bricks will be saved')
    parser.add_argument('--num_picks', type=int, default=1000,
                        help='The number of molecules to cluster')
    parser.add_argument('--draw_images', action='store_true', default=False,
                        help='Whether to draw images of Clustered molecules')
    return parser.parse_args()

if __name__ == '__main__':
    '''
    这个cluster好像有点问题
    '''
    args = parse_args()
    picker = MoleculePicker(args.process_dir,args.cluster_dir,args.num_picks,args.draw_images)
    picker.process_reactions()