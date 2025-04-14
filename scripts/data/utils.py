import warnings

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem 
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize

from data_utils import Reaction, ReactionSet


def strip_dummy_atoms(mol):
    dummy = Chem.MolFromSmiles('[*]')
    hydrogen = Chem.MolFromSmiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    return Chem.RemoveHs(mols[0])  

def get_morgen_fingerprint(smiles, nBits=4096):
    if smiles is None:
        warnings.warn("Received empty SMILES string")
        return np.zeros(nBits).reshape((-1, )).tolist()
    else:
        mol = Chem.MolFromSmiles(smiles)
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits)
        # features = np.zeros((1,))
        # DataStructs.ConvertToNumpyArray(fp_vec, features)
        # return features.reshape((-1, )).tolist()
        return fp_vec
def check_rxn_centre(smarts, smiles_list):
    patt = Chem.MolFromSmarts(smarts)
    temp_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        flag = mol.HasSubstructMatch(patt, useChirality=True) 
        if flag:
            matches = mol.GetSubstructMatches(patt)
            if len(matches) == 1:
                temp_smiles.append(s)
    return temp_smiles

def process_reac_file(rxns, building_blocks,rules_path,match_bb_path,rule_json_path):
    rxn_templates = []
    for reaction in rxns:
        try:
            rxn = Reaction(reaction)
            rxn.set_available_reactants(building_blocks)
            rxn_templates.append(rxn)
        except ValueError as e:
            print(e)
            continue
    
    fr_set = ReactionSet() 
    reac_file = open(rules_path, 'w')
    matched_ligs = set()     
    from tqdm import tqdm
    for r in tqdm(rxn_templates):
        template = r.reactant_template
        if r.num_reactant == 1:
            rt1 = r.available_reactants[0]
            t1 = template[0]
            r.available_reactants[0] = check_rxn_centre(t1, rt1)
            fr_set.rxns.append(r)
            reac_file.write(r.smirks + '\n')
        else:
            rt1, rt2 = r.available_reactants[0], r.available_reactants[1]
            if len(rt1) == 0 or len(rt2) == 0:
                continue
            else:
                [t1, t2] = template
                r.available_reactants[0] = check_rxn_centre(t1, rt1)
                r.available_reactants[1] = check_rxn_centre(t2, rt2)
                
                if len(r.available_reactants[0]) == 0 or len(r.available_reactants[1]) == 0:
                    continue
                else:
                    fr_set.rxns.append(r)
                    reac_file.write(r.smirks + '\n')
        
        for a_list in r.available_reactants:
            matched_ligs = matched_ligs | set(a_list)
            
    reac_file.close()  
    
    fr_set.save(rule_json_path)
    with open(match_bb_path, 'w') as f:
        for ml in matched_ligs:
            f.write(ml + '\n')
    
def get_reaction_mask(smi, rxns):
    if smi is None:
        return [0] * len(rxns)
    else:
        mol = Chem.MolFromSmiles(smi)
        reaction_mask = [int(rxn.is_reactant(mol)) for rxn in rxns]
        return reaction_mask

def plot_learning_curve(ep, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(ep, records, color='b', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file)

def get_available_list(tid, rxn):
    if tid == 0:
        available_list = rxn.available_reactants[1]
    elif tid == 1:
        available_list = rxn.available_reactants[0]
    else:
        available_list = []
        
    return available_list

def search_with_tanimoto(emb, avali_embs):
    emb = np.squeeze(emb)
    # sims = []
    # for aemb in avali_embs:
    #     tanimoto_smi = np.sum(emb * aemb) / (np.sum(emb) + np.sum(aemb) - np.sum(emb * aemb))
    #     sims.append(tanimoto_smi)
    #
    # sims = np.array(sims)
    # top_k = 9
    # top_simi_idx = sims.argsort()[::-1][0:top_k]

    emb_sum = np.sum(emb)  # 计算查询向量的元素和

    # 计算候选向量的元素和
    avali_embs_sum = np.sum(avali_embs, axis=1)

    # 计算查询向量和候选向量的内积
    dot_product = np.dot(avali_embs, emb)

    # 计算 Tanimoto 相似度
    tanimoto_sims = dot_product / (emb_sum + avali_embs_sum - dot_product)

    # 获取相似度最高的 top_k 个索引
    top_k = 9
    top_simi_idx = np.argsort(tanimoto_sims)[::-1][:top_k]

    return top_simi_idx

def create_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_properties(smi):
    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
    elif isinstance(smi, Chem.Mol):
        mol = smi
    
    from rdkit.Chem import Lipinski
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.Crippen import MolLogP
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    mw = round(ExactMolWt(mol), 2)
    logp = round(MolLogP(mol), 2)
    rotbonds = Lipinski.NumRotatableBonds(mol)
    rings = mol.GetRingInfo().NumRings()
    
    if mw <= 500 and hba <= 10 and hbd <= 5 and logp <= 5 and rotbonds <= 10 and rings <= 5:
        return True
    else:
        return False
    
def postprocessing(results, output):
    ref_smi = 'CC1=CC=C(NC(=O)CCCN2CCN(C/C=C/C3=CC=CC=C3)CC2)C(C)=C1'
    sims = []
    fp_ref = np.array(get_morgen_fingerprint(ref_smi))
    for res in results:
        fp_prod = np.array(get_morgen_fingerprint(res[-1][-2]))
        tanimoto_smi = np.sum(fp_ref * fp_prod) / (np.sum(fp_ref) + np.sum(fp_prod) - np.sum(fp_ref * fp_prod))
        sims.append(tanimoto_smi)
    sims = np.array(sims)
    rank_idx = sims.argsort()[::-1]
    
    filtered_res = []
    for i in range(len(results)):
        if get_properties(results[rank_idx[i]][-1][-2]):
            filtered_res.append(results[rank_idx[i]])
    
    import gzip, json
    with gzip.open(output, 'w') as f:
        f.write(json.dumps({"syn_paths": filtered_res}, indent=4).encode('utf-8'))
        print(output)

def standardize_smi(smiles,basicClean=True,clearCharge=True, clearFrag=True, canonTautomer=False, isomeric=False):
    try:
        clean_mol = Chem.MolFromSmiles(smiles)
        # clean_mol = Chem.SanitizeMol(clean_mol)
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