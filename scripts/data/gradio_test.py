import gradio as gr
from model_interface import *
from model_interface2 import *
from rdkit.Chem import Draw
from PIL import Image
import io


def smiles_to_image(smiles,img_size=(1000,1000)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol,size=img_size)    # Convert molecule to image
            return img
        else:
            return "Invalid SMILES"
    except Exception as e:
        return str(e)


def modified_smiles(smile):
    return re.sub(r'\[\d+\*\]', '[H]', smile)


# Get atom and bond indices for highlighting
def get_atom_and_bond_indices(mol, matches):
    atom_indices = [idx for match in matches for idx in match]
    bond_indices = [bond.GetIdx() for match in matches for bond in mol.GetBonds() if
                    bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match]
    return atom_indices, bond_indices


def draw_and_save_image(mol, atom_indices, bond_indices, colors, img_size=(1000, 1000),
                        filename="fragments_colored.png"):
    drawer = Draw.MolDraw2DCairo(img_size[0], img_size[1])

    # Create highlight color maps
    highlight_atom_colors = {i: colors[idx] for idx, atom_group in enumerate(atom_indices) for i in atom_group}
    highlight_bond_colors = {i: colors[idx] for idx, bond_group in enumerate(bond_indices) for i in bond_group}

    AllChem.Compute2DCoords(mol)

    # Draw molecule with highlighted atoms and bonds
    drawer.DrawMolecule(mol,
                        highlightAtoms=[i for group in atom_indices for i in group],
                        highlightBonds=[i for group in bond_indices for i in group],
                        highlightAtomColors=highlight_atom_colors,
                        highlightBondColors=highlight_bond_colors)

    drawer.FinishDrawing()
    img = drawer.GetDrawingText()
    return img


def fragment_to_image(smiles_input, cut_num, output_text1, output_text2, output_text3,break_choice, img_size=(1000, 1000)):
    try:
        if type(break_choice) is int:
            mol = Chem.MolFromSmiles(smiles_input)
            if cut_num == 2:
                # Modify SMILES to replace wildcards
                frag1, frag2, scaffold = modified_smiles(output_text1), modified_smiles(output_text2), modified_smiles(
                    output_text3)

                # Create substructure molecules
                substructure1 = Chem.MolFromSmiles(frag1)
                substructure2 = Chem.MolFromSmiles(frag2)
                substructure3 = Chem.MolFromSmiles(scaffold)

                matches1 = mol.GetSubstructMatches(substructure1)
                matches2 = mol.GetSubstructMatches(substructure2)
                matches3 = mol.GetSubstructMatches(substructure3)

                # Define highlight colors (RGBA with transparency)
                colors = [(0.5, 1.0, 0.5, 0.7),  # 氮原子：浅绿色 (Light Green) 半透明
                          (0.8, 0.6, 1.0, 0.7),  # 碳原子：浅紫色 (Light Purple) 半透明
                          (1.0, 1.0, 0.0, 0.7),  # 氧原子：亮黄色 (Yellow) 半透明
                          ]

                # Get atom/bond indices
                atom_indices1, bond_indices1 = get_atom_and_bond_indices(mol, matches1)
                atom_indices2, bond_indices2 = get_atom_and_bond_indices(mol, matches2)
                atom_indices3, bond_indices3 = get_atom_and_bond_indices(mol, matches3)

                img_bytes = draw_and_save_image(mol, [atom_indices1, atom_indices2, atom_indices3],
                                                [bond_indices1, bond_indices2, bond_indices3], colors, img_size)
            else:
                frag1, frag2= modified_smiles(output_text1), modified_smiles(output_text2)

                substructure1 = Chem.MolFromSmiles(frag1)
                substructure2 = Chem.MolFromSmiles(frag2)

                matches1 = mol.GetSubstructMatches(substructure1)
                matches2 = mol.GetSubstructMatches(substructure2)

                colors = [(0.5, 1.0, 0.5, 0.7),  # 氮原子：浅绿色 (Light Green) 半透明
                          (0.8, 0.6, 1.0, 0.7)]  # 碳原子：浅紫色 (Light Purple) 半透明

                atom_indices1, bond_indices1 = get_atom_and_bond_indices(mol, matches1)
                atom_indices2, bond_indices2 = get_atom_and_bond_indices(mol, matches2)


                img_bytes = draw_and_save_image(mol, [atom_indices1, atom_indices2],
                                                [bond_indices1, bond_indices2], colors, img_size)
            img = Image.open(io.BytesIO(img_bytes))

            # img.save("fragments_colored.png",dpi=(1200,1200))
            return img

        else:
            return smiles_to_image(smiles_input)

    except Exception as e:
        print(e)
        return str(e)


def output_text_change(smiles, min_fragment_size, cut_num, index):
    if smiles:
        combinations = Bricap_break(smiles, min_fragment_size, cut_num)
        if type(index) is int:
            if cut_num == 1:
                print(combinations)
                return [combinations[int(index) - 1][0]] + ['scaffold'] + [combinations[int(index) - 1][1]]+ [combinations[int(index) - 1]] + [combinations]
            else:
                # The last two are output to the hidden box
                return combinations[int(index) - 1] + [combinations[int(index) - 1]] + [combinations]

        return ["Please select a combination from the dropdown"] * 5
    else:
        return ["No SMILES input"] * 5


def cut_num_update(cut_num):
    if cut_num == 1:
        return gr.update(visible=True),gr.update(visible=True), gr.update(visible=False),gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=True),gr.update(visible=True), gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)

def cut_num_update_frag(cut_num):
    if cut_num == 1:
        return gr.update(visible=True),gr.update(visible=True), gr.update(visible=False),gr.update(visible=True), gr.update(visible=False),gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True),gr.update(visible=True), gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True), gr.update(visible=False)

def update_dropdown(smiles, min_fragment_size, cut_num):
    if smiles:
        combinations = Bricap_break(smiles, min_fragment_size, cut_num)
        print(combinations)
        break_combinations = range(1,len(combinations)+1)
        return gr.update(choices=break_combinations, value=combinations[0])  # Dynamically update the combination


with gr.Blocks(theme='compact',css="./web_style/style.css") as demo:
    with gr.Tab(label="Chemical Space Search",elem_classes='gr-tab-item'):
        with gr.Row():
            # Create an input box to accept SMILES input
            # smiles_input = gr.Textbox(lines=5, label="SMILES", elem_classes="input", scale=4)
            with gr.Column():
                with gr.Row():
                    smiles_input = gr.Textbox(label="SMILES", elem_classes="input-box")

                gr.Markdown("<h3>Bricap Parameters</h3>", elem_classes="section-title")
                with gr.Row():
                    minFragmentSize = gr.Slider(minimum=0, maximum=10, step=1, label="Filter fragment size",
                                                value=4)
                    cut_num = gr.Slider(minimum=1, maximum=2, step=1, label="Fragmentation level", value=2)

                break_choice = gr.Dropdown(choices=None, label="Break the ligand",value=None)

                with gr.Row():
                    output_text1 = gr.Textbox(label="Fragment1",value="No SMILES input")
                    output_text2 = gr.Textbox(label="Fragment2", value="No SMILES input")
                    output_text3 = gr.Textbox(label="scaffold", value="No SMILES input", visible=True)
                    output_text4 = gr.Textbox(label="combination", value="No SMILES input", visible=False)
                    output_text5 = gr.Textbox(label="break_list", value="No SMILES input", visible=False)


                smiles_input.submit(update_dropdown, inputs=[smiles_input, minFragmentSize, cut_num],
                                    outputs=break_choice)

                break_choice.change(output_text_change, inputs=[smiles_input, minFragmentSize, cut_num, break_choice],
                                    outputs=[output_text1, output_text3, output_text2,output_text4,output_text5])

            # Output area to display the generated molecule image

            smile_image = gr.Image(label='Molecule Image'
                                   ,show_download_button=False,container=False)

            # Define the image update operation triggered by fragment combination
            output_text4.change(fragment_to_image, inputs=[smiles_input,cut_num,output_text1,output_text2,output_text3,break_choice]
                                ,outputs=smile_image)

            with gr.Row():
                BB_data = gr.Dropdown(
                    choices=["Enamine stock", "Enamine BB Catalog", "Mcule"],
                    label="Choose BB Library",
                    value="Enamine BB Catalog",
                    elem_classes = "dropdown-box"
                )

                Predict = gr.Button(value="Search", min_width=100, variant="primary",elem_classes="blue-gradient-button")

                with gr.Column():
                    fingerprint = gr.Radio(choices=["Morgan", "MACCS","Pharm2D"], label="Fingerprint type", value='Morgan')
                    gr.Markdown("<h3>Similarity search threshold<h3>", elem_classes="section-title")

                    Endpoint_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Endpoint threshold"
                                           ,value=0.6)
                    Scaffold_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Scaffold threshold",
                                           value=0.6,visible=True)
                    chemprop_checkboxgroup = gr.CheckboxGroup(
                        ['MW', 'HAC','slogP', 'HBA', 'HBD', 'TPSA', 'RotBonds', 'QED', 'SAscore']
                        , label='Products calculate property',value=['MW', 'HAC', 'slogP', 'HBA', 'HBD', 'TPSA', 'RotBonds', 'QED', 'SAscore'])
                cut_num.change(fn=cut_num_update,
                               inputs=cut_num,
                               outputs=[output_text1, output_text2, output_text3,Endpoint_threshold, Scaffold_threshold])

        gr.Markdown("<h3>Results(Property distribution and products)</h3>", elem_classes="section-title")
        with gr.Row():
            property_image = gr.Plot(label='Product property image')
            table_result = gr.DataFrame(headers= ['Brick','Brick_ID','Scaffold','Scaffold_ID','Product','Property']
                                        , col_count=6,elem_classes="result_table",scale=2)

            Predict.click(fn=Chemical_Space_Search, inputs=[smiles_input,BB_data,fingerprint,Endpoint_threshold
                ,Scaffold_threshold,output_text5,chemprop_checkboxgroup], outputs=[table_result,property_image])

    with gr.Tab(label="Chemical Fragment Replacement", elem_classes='gr-tab-item-frags'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    smiles_input = gr.Textbox(label="SMILES", elem_classes="input-box")

                gr.Markdown("<h3>Bricap Parameters</h3>", elem_classes="section-title")
                with gr.Row():
                    minFragmentSize = gr.Slider(minimum=0, maximum=10, step=1, label="Filter fragment size",
                                                value=4)
                    cut_num = gr.Slider(minimum=1, maximum=2, step=1, label="Fragmentation level", value=2)

                break_choice = gr.Dropdown(choices=None, label="Break the ligand", value=None)

                with gr.Row():
                    output_text1 = gr.Textbox(label="Fragment1", value="No SMILES input")
                    output_text2 = gr.Textbox(label="Fragment2", value="No SMILES input")
                    output_text3 = gr.Textbox(label="scaffold", value="No SMILES input", visible=True)
                    output_text4 = gr.Textbox(label="combination", value="No SMILES input", visible=False)
                    output_text5 = gr.Textbox(label="break_list", value="No SMILES input", visible=False)

                smiles_input.submit(update_dropdown, inputs=[smiles_input, minFragmentSize, cut_num],
                                    outputs=break_choice)

                break_choice.change(output_text_change, inputs=[smiles_input, minFragmentSize, cut_num, break_choice],
                                    outputs=[output_text1, output_text3, output_text2, output_text4, output_text5])

            # Output area to display the generated molecule image
            smile_image = gr.Image(label='Molecule Image'
                                   , show_download_button=False, container=False)

            # Define the image update operation triggered by fragment combination
            output_text4.change(fragment_to_image,
                                inputs=[smiles_input, cut_num,output_text1, output_text2, output_text3, break_choice]
                                , outputs=smile_image)

            with gr.Row():
                BB_data = gr.Dropdown(
                    choices=["Enamine stock", "Enamine BB Catalog", "Mcule"],
                    label="Choose BB Library",
                    value="Enamine BB Catalog",
                    elem_classes="dropdown-box"
                )

                Predict = gr.Button(value="Replace", min_width=100, variant="primary", elem_classes="purple-gradient-button")

                with gr.Column():
                    fingerprint = gr.Radio(choices=["Morgan", "MACCS", "Pharm2D"], label="Fingerprint type",
                                           value='Morgan')
                    fragment1 = gr.Radio(choices=["Fragment1", "Scaffold", "Fragment2"], label="Keep fragment")
                    fragment2 = gr.Radio(choices=["Fragment1", "Fragment2"], label="Keep fragment",visible=False)

                    First_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="First threshold"
                                                   , value=0.6)
                    Second_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Second threshold",
                                                   value=0.6, visible=True)
                    chemprop_checkboxgroup = gr.CheckboxGroup(
                        ['MW', 'HAC', 'slogP', 'HBA', 'HBD', 'TPSA', 'RotBonds', 'QED', 'SAscore']
                        , label='Products calculate property',
                        value=['MW', 'HAC', 'slogP', 'HBA', 'HBD', 'TPSA', 'RotBonds', 'QED', 'SAscore'])

                cut_num.change(fn=cut_num_update_frag,
                               inputs=cut_num,
                               outputs=[output_text1, output_text2, output_text3, First_threshold
                                   , Second_threshold, fragment1, fragment2])

        gr.Markdown("<h3>Results(Property distribution and products)</h3>", elem_classes="section-title")
        with gr.Row():
            property_image = gr.Plot(label='Product property image')
            table_result = gr.DataFrame(headers=['Brick', 'Brick_ID', 'Scaffold', 'Scaffold_ID', 'Product', 'Property']
                                        , col_count=6, elem_classes="frag_result_table", scale=2)
            Predict.click(fn=Fragment_Space_Search, inputs=[smiles_input, BB_data, fingerprint, First_threshold
                , Second_threshold, output_text5, chemprop_checkboxgroup, fragment1, fragment2 , break_choice], outputs=[table_result, property_image])


demo.launch(server_port = 6860)
# demo.launch(server_port=0)
# demo.launch(server_name="0.0.0.0", server_port=7860)


