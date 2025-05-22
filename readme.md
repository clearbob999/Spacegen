# 🧬SpaceGen: Dynamic Chemical Space Exploration Accelerates Drug Design via Reinforcement Learning and Assembly Optimization

![模型总括图](F:/yanerxia/typora/images/%E6%A8%A1%E5%9E%8B%E6%80%BB%E6%8B%AC%E5%9B%BE.png)

## 📚Content

- [Overview](#Overview)
- [Software Requirements](#Software Requirements)
- [Installation Guide](#Installation Guide)
- [Data](#Data)
- [SpaceGen](#SpaceGen)
- [SpaceGen(ligand)](#SpaceGen(ligand))

## 🚀Overview

This study presents SpaceGen, an innovative and scalable framework for ultra-large chemical space construction and intelligent molecule generation. By integrating diverse building block sources, refining reaction rule mapping, and incorporating reinforcement learning strategies, SpaceGen addresses limitations in current virtual libraries and generative models. The method supports compact storage (<100 MB), enables rapid retrieval, and ensures synthetic feasibility. Core innovations include BRICAP-based fragmentation for efficient reconstruction, TD3-guided molecular generation, and a hybrid virtual screening pipeline combining ligand- and structure-based approaches. Validated on three drug targets, SpaceGen successfully produced nanomolar lead compounds, demonstrating high efficiency, targetability, and practical utility in modern drug discovery.

## 💻 Software Requirements

This package is supported for *Linux* and *Windows*. The package has been tested on the following systems:

- Windows: Windows 11 24H2
- Linux: Ubuntu 22.04

### Software requirements

- Python == 3.9
- pytorch >= 1.12.0
- openbabel == 2.4.1
- RDKit == 2020.09.5
- [qvina02](https://qvina.github.io/)(for linux)
- openbabel >= 3.1.1

## ⚙️Installation Guide

```
git clone https://github.com/clearbob999/Spacegen.git
cd Spacegen
conda env create -f environment.yaml
conda activate Spacegen
```

## 📦Data

### Data Preparation

Some of the initial data files need to be requested from Enamine. You can apply for access to the building block catalog [here][https://enamine.net/building-blocks/building-blocks-catalog]

```
cd Data
├── 1_brickdata/
    ├── BBstock.sdf          # Building blocks from Enamine globalstock
    ├── BBcatalog.sdf        # Building blocks from Enamine building block catalog
    └── MucleBB.sdf     	# Building blocks from mcule https://mcule.com/database/
```

### Data Preprocessing

After placing the required SDF files into `Data/1_brickdata/`, please follow these steps to preprocess and prepare the data for molecular generation and screening:

```python
cd script/data

# Step 1: Preprocess building blocks from the input SDF file:
# remove small fragments, normalize charges, chirality, and tautomers.
# Extract classification information and save as a CSV file.
python brick_extract_1.py \
  --input_path ../../Data/1_brickdata/Enamine_BB.sdf \
  --fingerprint morgan \
  --rule_path ../../Data/2_rules/rules.xlsx \
  --process_dir ../../Data/1_brickdata/brick_process

# Step 2: Map building blocks to reaction rules using the rule file.
# Serialize building block fingerprints and save in feather format.
python bricks_to_rules_2.py \
  --input_path ../../Data/1_brickdata/Enamine_BB.sdf \
  --fingerprint morgan \
  --rule_path ../../Data/2_rules/rules.xlsx \
  --process_dir ../../Data/1_brickdata/brick_process

# Step 3 (Optional): Cluster building blocks to reduce redundancy.
# Select a representative subset (e.g., 1000 blocks) for efficient sampling.
python brick_cluster_3.py \
  --process_dir ../../Data/1_brickdata/brick_process \
  --cluster_dir ../../Data/1_brickdata/brick_cluster \
  --num_picks 1000

# Step 4 (Optional): Construct the virtual chemical space by combinatorially
# assembling clustered building blocks using the mapped reaction rules.
# Output the chemical space in CSV format.
python reaction_multiprocess_4.py \
  --rule_path ../../Data/2_rules/rules.xlsx \
  cluster_dir ../../Data/1_brickdata/brick_cluster \
  --output ../../Data/3_output/chem_space \
  --format csv
```

## 🔧 SpaceGen Module

This module enables ligand-guided structure generation using a combination of molecular fragmentation, building block mapping, and reinforcement learning-based molecular assembly. It is particularly useful for target-based molecule design, especially when starting from known ligands or bioactive compounds.

#### Requirements

- SMILES string of a ligand (from ChEMBL, PubChem, or other bioactivity databases)
- Preprocessed building block library (e.g., Enamine BB Catalog, Enamine Stock, or Mcule BBs)
- Reaction rules file (provided in `Data/2_rules`)

```python
python inhibitor_data.py \
  -BB_data "Enamine BB Catalog" \
  -minFragmentSize 4 \
  -cut_num 2 \
  -threshold 0.6 \
  -fingerprint Morgan \
  --feature 256
```



### ▶️ Run 

The reinforcement learning (RL) module of SpaceGen is designed to iteratively generate target-specific candidate molecules guided by reward functions such as docking scores, similarity, or synthetic accessibility. This module uses a reaction rule-based block assembly strategy combined with reward optimization to generate novel structures with desired properties.

```python
python main.py \
  --match_brick_npy data/{target}_matched_bbs_emb_256.npy \
  --match_brick_txt data/{target}_matched_bbs.txt \
  --match_rules data/{target}_rxn_filter.txt \
  --receptor_file data/{target}.pdbqt \
  --rule_json_path data/{target}_data_for_reaction_filtered.json.gz
```



## 🔬 SpaceGen (Ligand Module)

The **Ligand Module** in SpaceGen enables fragmentation and matched building block search for a given ligand. It offers two usage modes: a **command-line interface (CLI)** for batch or scriptable usage, and a **visual interface** via Gradio for interactive exploration.

------

### Option 1: Command-Line Interface (CLI)

You can run the ligand fragmentation and building block search via CLI:

```python
python break_ligand_5.py \
  --input FC1=CC(OC)=C(C2=C(Cl)C=NC(NC3=NN(C \
  --BB_data 'Enamine BB Catalog' \
  --minFragmentSize 4 \
  --cut_num 3 \
  --threshold 0.2
```

###  Option 2: Gradio Web Interface

You can also explore ligand fragmentation and matched building blocks visually using the Gradio web interface:

```python
python gradio.py
```

![gradio界面](F:/yanerxia/typora/images/gradio%E7%95%8C%E9%9D%A2.png)

## License

This project is covered under the MIT License.