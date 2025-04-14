import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
import unittest

# These are the definitions that will be applied to fragment molecules:
reactionDefs_raw = (
  "[#7;+0;D2,D3:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>*[#7:1].[#7:2]*",  # urea  0
  "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>*[C:1]=[O:2].*[#7:3]",  # amide  1
  "[C:1](=!@[O:2])!@[O;+0:3]>>*[C:1]=[O:2].[O:3]*",  # ester  2
  "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>*[*:1].[*:2]*",  # amines  3
  # "[N;!D1](!@[*:1])!@[*:2]>>*[*:1].[*:2]*", # amines

  # again: what about aromatics?
  "[#7;R;D3;+0:1]-!@[*:2]>>*[#7:1].[*:2]*",  # cyclic amines  4
  "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1]*.*[#6:2]",  # ether  5
  "[C:1]=!@[C:2]>>[C:1]*.*[C:2]",  # olefin  6
  "[n;+0:1]-!@[C:2]>>[n:1]*.[C:2]*",  # aromatic nitrogen - aliphatic carbon  7
  "[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[O:3]=[C:4]-[N:1]*.[C:2]*",  # lactam nitrogen - aliphatic carbon  8
  "[c:1]-!@[c:2]>>[c:1]*.*[c:2]",  # aromatic carbon - aromatic carbon  9
  # aromatic nitrogen - aromatic carbon *NOTE* this is not part of the standard recap set.
  "[n;+0:1]-!@[c:2]>>[n:1]*.*[c:2]",   # 10
  "[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[#7:1]*.*[S:2](=[O:3])=[O:4]",  # sulphonamide   11
)

reactionDefs = (
  # "[#6:5][C:1](=[O:2])-[N:3]-[#6:4]>>[#6:5][C:1](=[O:2])[OH].[#6;D4:4][N;D3:3]", # amide  0
  # "[#7;+0;D2,D3;!$(N[C,S]=[S,O,N]):1]-!@[c:2]>>[N:1]-[21*].[c:2]-[22*]", # aromati nitrogen - aromatic carbon  buchwald 10
  "[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>[C:1](=[O:2])-[1*].[#7:3]-[2*]", # amide  0
  "[#6:5][C:1](=!@[O:2])-[O;+0:3]-[#6:4]>>[#6:5][C:1](=[O:2])-[3*].[#6;D4:4]-[4*]", # ester  1    # 出现3是因为OH是后续加的
  "[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[#7:1]-[5*].[6*]-[S:2](=[O:3])=[O:4]", # sulphonamide  2
  "[N;!$(N-C=[#7,#8,#15,#16]):1](-!@[*:2])(-!@[*:3])-!@[*:4]>>[*:4]-[7*].[8*]-[N:1](-[*:2])(-[*:3])", # amines 3
  "[#7;+0;D2:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>[#7:1]-[9*].[10*]-[#7:2]", # urea  4
  "[#6:1]-!@[O;+0]-!@[#6:2]>>[#6:1]-[11*].[#6:2]-[12*]", # ether  5   Williamson_ether
  "[C:1]=!@[C:2]>>[C:1]-[13*].[14*]-[C:2]", # olefin  6   witting 6
  "[n;+0:1]-!@[C:2]>>[n:1]-[15*].[C:2]-[16*]", # aromatic nitrogen - aliphatic carbon  7  N-wanjihua
  "[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[O:3]=[C:4]-[N:1]-[17*].[C:2]-[18*]", # lactam nitrogen - aliphatic carbon  8
  "[c:1]-!@[c:2]>>[c:1]-[19*].[c:2]-[20*]", # aromatic carbon - aromatic carbon  9 suzuki
  "[#7;+0;D2,D3;!$(N[C,S]=[S,O,N]):1]-!@[c:2]>>[N:1]-[21*].[c:2]-[22*]", # aromati nitrogen - aromatic carbon  buchwald 10

  "[c:1]1:[n:2]:[nH]:[c:3]:[n]@1>>[C:1]#[N:2].[OH][C:3](=[O])", # 1,2,4-triazole_carboxylic-acid_ester 1,2,4-三氮唑合成（酸酯） 11
  "[#6:6][c:5]1[n:4][o:3][c:2]([#6:1])n1>>[#6:6][C:5]#[#7;D1:4].[#6:1][C:2](=[OD1:3])[OH1]", # zahuan Oxadiazole  12 噁唑烷二酮
  "[c:1]1:[c:2]:n(-[C:3]):n:n@1>>[C:1]#[CH1:2].[C:3]-[#17]", # Huisgen_Cu 三唑环裂解 13  [*;#17,#35,#53,OH1]
  "[c:1]1:[c:2]:n:n:n(-[C:3])@1>>[C:1]#[CH1:2].[C:3]-[#17]", # Huisgen_Ru 三唑环裂解 14  [*;#17,#35,#53,OH1]
  "[c:3]12:[c:1](:[c:10]:[c:9]:[c:8]:[c:7]:1):[n:2]:[c:5](-[#6:6])=[n:4]@2>>[c:1]1(-[N:2]):[c:3](-[N:4]):[c:7]:[c:8]:[c:9]:[c:10]@1.[#6:6]-[CH1:5](=[OD1])", # Benzimidazole_derivatives_aldehyde 苯并咪唑(醛) 15
  "[c:3]12:[c:1](:[c:10]:[c:9]:[c:8]:[c:7]:1):[o:2]:[c:5](-[#6:6]):[n:4]@2>>[c:1]1(-[O:2]):[c:3](-[N:4]):[c:10]:[c:9]:[c:8]:[c:7]@1.[#6:6]-[C:5](=[OD1])-[O]", # Benzoxazole_carboxylic-acid 苯并噁唑(酸) 16
  "[c:1]12:[c:2](-[CH2:7]-[CH2:8]-[NH1:9]-[C:10]-2(-[#6:11])):[c:3]:[c:4]:[c:5]:[c:6]:1>>[c:1]1:[c:2](-[C:7]-[C:8]-[N:9]):[c:3]:[c:4]:[c:5]:[c:6]:1.[#6:11]-[C:10]=[OD1]", # Pictet-Spengler 四氢异喹啉合成 17
  "[c:1]12:[c:2](:[c:9]:[c:8]:[nH:7]:2):[c:3]:[c:4]:[c:5]:[c:6]:1>>[c:1]1(-[N:7]-[N]):[c:2]:[c:3]:[c:4]:[c:5]:[c:6]:1.[C:9]-[C:8](=[O])", # Fischer indole 人名反应（合成吲哚） 18
  "[c:1]12:[c:2](:[c:9]:[c:8]:[c:7]:[c:6]:2):[o:3]:[c:4]:[cH:5]@1>>[Br]-[c:1]2:[c:2](:[c:9]:[c:8]:[c:7]:[c:6]:2)-[O:3].[#6:5]#[#6:4]", # Benzofuran 苯并呋喃合成 19
  "[c:1]12:[c:2](:[c:11]:[c:10]:[c:9]:[c:8]:2):[c:3](=[O:4]):[n:5]:[c:6]:[n:7]@1>>[#7:7][c:1]2:[c:2](:[c:11]:[c:10]:[c:9]:[c:8]:2)-[#6:3](=[OD1:4])-[O].[N:5]-[C:6]=[OD1]", # Niementowski_quinazoline 邻氨基苯甲酸与酰胺反
  "[c:1]12:[c:2](:[c:10]:[c:9]:[c:8]:[c:7]:2):[s:3]:[c:4]([#6:6]):[n:5]:1>>[c:1](-[#7:5])2:[c:2](:[c:10]:[c:9]:[c:8]:[c:7]:2)-[S:3].[#6:6]-[C:4](=[OD1])", # Benzothiazole 苯并噻唑合成（醛） 21
  "[c:1]2(-[#6:6]):[n:2]:[c:3]:[s:4][c:5]([#6:7]):2>>[#6:6]-[C;R0:1](=[OD1])-[CH1;R0:5](-[#6:7])-[#17].[NH2:2]-[C:3]=[SD1:4]", # Thiazole 噻唑合成 22
  "[c:1]12:[c:2](:[n:7]:[c:8]:[c:9]:[c:10]:2):[c:3]:[c:4]:[c:5]:[c:6]:1>>[c:1](-[C:10]=[O])1:[c:2](-[NH2:7]):[c:3]:[c:4]:[c:5]:[c:6]:1.[#6:8](=[OD1])-[#6:9]", # Friedlaender_quinoline 人名反应（合成喹啉）23 解决
  "[c:1]2:[c:2]:[c:3]:[n:4]:[n:5]:[c:6](=[O:7])@2>>[c:1](-[C:6](=[O:7])[OH]):[c:2]-[C:3]=[O].[N;H2:4]-[N;H1:5]", # Phthalazinone 二氮杂萘酮合成 24
  "[c:1]12:[c:2](:[c:9]:[c:8]:[c:7]:[c:6]:1):[s:3]:[c:4]:[cH:5]@2>>[Br]-[c:1]1:[c:2](-[S:3]-[CH3]):[c:9]:[c:8]:[c:7]:[c:6]:1.[C:4]#[C:5]", # Benzothiophene 苯并噻吩合成 25
  "[O:6]1-[c:5]:[c:1]-[C:2](=[OD1:3])-[C:4]-[C:7]-1>>[c:1](-[C;$(C-c1ccccc1):2](=[OD1:3])-[CH3:4]):[c:5](-[OH1:6]).[C;$(C1-[CH2]-[CH2]-[N,C]-[CH2]-[CH2]-1):7](=[OD1])", # Spirochromanone 螺环香兰酮合成 26
)
#[c:1]12:[c:2](-[CH2:7]-[CH2:8]-[NH1:9]-[C:10]-2):[c:3]:[c:4]:[c:5]:[c:6]:1   四氢喹唑啉超级经典的结构写法
#[cH1:1]1:[c:2](-[CH2:7]-[CH2:8]-[NH2:9]):[c:3]:[c:4]:[c:5]:[c:6]:1    这是苯乙胺的结构写法
reactions = tuple([Reactions.ReactionFromSmarts(x) for x in reactionDefs])


class RecapHierarchyNode(object):
    """ This class is used to hold the Recap hiearchy
    """
    mol = None
    children = None
    parents = None
    smiles = None

    def __init__(self, mol):
        self.mol = mol
        self.children = {}
        self.parents = {}

    def GetAllChildren(self):
        " returns a dictionary, keyed by SMILES, of children "
        res = {}
        for smi, child in self.children.items():
            res[smi] = child
            child._gacRecurse(res, terminalOnly=False)
        return res

    def GetLeaves(self):
        " returns a dictionary, keyed by SMILES, of leaf (terminal) nodes "
        res = {}
        for smi, child in self.children.items():
            if not len(child.children):
                res[smi] = child
            else:
                child._gacRecurse(res, terminalOnly=True)
        return res

    def getUltimateParents(self):
        """ returns all the nodes in the hierarchy tree that contain this
            node as a child
        """
        if not self.parents:
            res = [self]
        else:
            res = []
            for p in self.parents.values():
                for uP in p.getUltimateParents():
                    if uP not in res:
                        res.append(uP)
        return res

    def _gacRecurse(self, res, terminalOnly=False):
        for smi, child in self.children.items():
            if not terminalOnly or not len(child.children):
                res[smi] = child
            child._gacRecurse(res, terminalOnly=terminalOnly)

    def __del__(self):
        self.children = {}
        self.parents = {}
        self.mol = None


def RecapDecompose(mol, allNodes=None, minFragmentSize=0, onlyUseReactions=None):
    """ returns the recap decomposition for a molecule """
    mSmi = Chem.MolToSmiles(mol, 1, kekuleSmiles=1)

    if allNodes is None:
        allNodes = {}
    if mSmi in allNodes:
        return allNodes[mSmi]

    res = RecapHierarchyNode(mol)
    res.smiles = mSmi
    activePool = {mSmi: res}
    allNodes[mSmi] = res
    while activePool:
        nSmi = next(iter(activePool))
        node = activePool.pop(nSmi)
        if not node.mol:
            continue
        for rxnIdx, reaction in enumerate(reactions):
            if onlyUseReactions and rxnIdx not in onlyUseReactions:
                continue
            # print '  .',nSmi
            # print '         !!!!',rxnIdx,nSmi,reactionDefs[rxnIdx]
            ps = reaction.RunReactants((node.mol, ))
            # print '    ',len(ps)
            if ps:
                for prodSeq in ps:
                    seqOk = True
                    # we want to disqualify small fragments, so sort the product sequence by size
                    # and then look for "forbidden" fragments
                    tSeq = [(prod.GetNumAtoms(onlyExplicit=True), idx)
                            for idx, prod in enumerate(prodSeq)]
                    tSeq.sort()
                    ts = [(x, prodSeq[y]) for x, y in tSeq]
                    prodSeq = ts
                    for nats, prod in prodSeq:
                        try:
                            Chem.SanitizeMol(prod)
                        except Exception:
                            continue
                        pSmi = Chem.MolToSmiles(prod, 1, kekuleSmiles=1)
                        if minFragmentSize > 0:
                            nDummies = pSmi.count('*')
                            if nats - nDummies < minFragmentSize:
                                seqOk = False
                                break
                        # don't forget after replacing dummy atoms to remove any empty
                        # branches:
                        elif pSmi.replace('*', '').replace('()', '') in ('', 'C', 'CC', 'CCC'):  #除去甲基，乙基，丙基这种
                            seqOk = False
                            break
                        prod.pSmi = pSmi
                    if seqOk:
                        for nats, prod in prodSeq:
                            pSmi = prod.pSmi
                            # print '\t',nats,pSmi
                            if pSmi not in allNodes:
                                pNode = RecapHierarchyNode(prod)
                                pNode.smiles = pSmi
                                pNode.parents[nSmi] = weakref.proxy(node)
                                node.children[pSmi] = pNode
                                activePool[pSmi] = pNode
                                allNodes[pSmi] = pNode
                            else:
                                pNode = allNodes[pSmi]
                                pNode.parents[nSmi] = weakref.proxy(node)
                                node.children[pSmi] = pNode
                        # print '                >>an:',allNodes.keys()
    return res

# ------- ------- ------- ------- ------- ------- ------- -------
# Begin testing code
def get_leaves(recap_decomp, n=1):
    for child in recap_decomp.children.values():
        print('\t' * n + '第%d层'%n +child.smiles)
        if child.children:  #进一步检查碎片
            get_leaves(child, n=n + 1)

def get_recap_tree(mol,onlyUseReactions):
    recap = RecapDecompose(mol,onlyUseReactions=onlyUseReactions)
    print(Chem.MolToSmiles(mol))
    get_leaves(recap)

def testfunction(smile,onlyUseReactions):
    m = Chem.MolFromSmiles(smile)
    print(Chem.MolToSmiles(m))
    res = RecapDecompose(m,onlyUseReactions=onlyUseReactions)
    children = res.GetLeaves().keys()
    allchildren = res.GetAllChildren().keys()
    Leaves = res.GetLeaves().keys()
    get_recap_tree(m,onlyUseReactions=onlyUseReactions)
    return children,allchildren,Leaves

class TestCase(unittest.TestCase):

    def testAmideRxn(self):
        m = Chem.MolFromSmiles('O=C1CC(C)C(C(C)C)CN1')
        # m = Chem.MolFromMol2File('.\\.\\database\\scPDB_ligand\\10mh_1_ligand.mol2')
        print(Chem.MolToSmiles(m))
        # print(BRICS.BRICSDecompose(m))
        res = RecapDecompose(m,onlyUseReactions=[0])
        children2 = res.GetLeaves().keys()
        print('Cyclic amide subnode:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('All subnodes of cyclic amides:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Leaf nodes of cyclic amides:{}'.format(Leaves2))
        get_recap_tree(m,[0])

        m = Chem.MolFromSmiles('NC(c1c(N2CCC3(CC3)CC2)cccc1)=O')
        # m = Chem.MolFromMol2File('.\\.\\database\\scPDB_ligand\\10mh_1_ligand.mol2')
        print(Chem.MolToSmiles(m))
        # print(BRICS.BRICSDecompose(m))
        res = RecapDecompose(m, onlyUseReactions=[0])
        children2 = res.GetLeaves().keys()
        print('terminal amides subnode:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('All subnodes of terminal amides:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Leaf nodes of terminal amides:{}'.format(Leaves2))
        get_recap_tree(m,[0])

        m = Chem.MolFromSmiles('Cc1ncnc(NC(c2c(N3CCC4(CC4)CC3)cc(N)cc2)=O)c1')
        # m = Chem.MolFromMol2File('.\\.\\database\\scPDB_ligand\\10mh_1_ligand.mol2')
        print(Chem.MolToSmiles(m))
        # print(BRICS.BRICSDecompose(m))
        res = RecapDecompose(m, onlyUseReactions=[0])
        children2 = res.GetLeaves().keys()
        print('Amide subnode:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('All subnodes of Amide:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Leaf nodes of Amide:{}'.format(Leaves2))
        get_recap_tree(m, [0])

    def testEsterRxn(self):
        m = Chem.MolFromSmiles('C1CC1C(=O)OC1OC1')
        print(Chem.MolToSmiles(m))
        res = RecapDecompose(m, onlyUseReactions=[1])
        children2 = res.GetLeaves().keys()
        print('Ester subnode:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('All subnodes of Ester:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Leaf nodes of Ester:{}'.format(Leaves2))
        get_recap_tree(m, [1])

        m = Chem.MolFromSmiles('C1CC1C(=O)CC1OC1')
        print(Chem.MolToSmiles(m))
        res = RecapDecompose(m, onlyUseReactions=[1])
        children2 = res.GetLeaves().keys()
        print('Ester subnode:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('All subnodes of Ester:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Leaf nodes of Ester:{}'.format(Leaves2))
        get_recap_tree(m, [1])

        m = Chem.MolFromSmiles('C1CCC(=O)OC1')
        print(Chem.MolToSmiles(m))
        res = RecapDecompose(m, onlyUseReactions=[1])
        children2 = res.GetLeaves().keys()
        print('Ester cyclic ester subnodes:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('Ester cyclic ester all subnodes:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Leaf nodes of Ester cyclic ester:{}'.format(Leaves2))
        get_recap_tree(m, [1])


    def testSulfonamideRxn(self):
        m = Chem.MolFromSmiles('COC1=CC=C(C=C1)CCNC(C2=CC=CC=C2NS(=O)(C3=C(NN=C3)C)=O)=O')
        print(Chem.MolToSmiles(m))
        res = RecapDecompose(m, onlyUseReactions=[2])
        children2 = res.GetLeaves().keys()
        print('Sulfonamide磺酰胺subnodes:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('Sulfonamide磺酰胺all subnodes:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Sulfonamide磺酰胺Leaf nodes:{}'.format(Leaves2))
        get_recap_tree(m, [2])

        m = Chem.MolFromSmiles('CCCNS(=O)(=O)CC')
        print(Chem.MolToSmiles(m))
        res = RecapDecompose(m, onlyUseReactions=[2])
        children2 = res.GetLeaves().keys()
        print('Sulfonamide磺酰仲胺subnodes:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('Sulfonamide磺酰仲胺all subnodes:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Sulfonamide磺酰仲胺Leaf nodes:{}'.format(Leaves2))
        get_recap_tree(m, [2])

        m = Chem.MolFromSmiles('c1cccn1S(=O)(=O)CC')
        print(Chem.MolToSmiles(m))
        res = RecapDecompose(m, onlyUseReactions=[2])
        children2 = res.GetLeaves().keys()
        print('Sulfonamide磺酰叔胺subnodes:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('Sulfonamide磺酰叔胺all subnodes:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Sulfonamide磺酰叔胺Leaf nodes:{}'.format(Leaves2))
        get_recap_tree(m, [2])

        m = Chem.MolFromSmiles('C1CNS(=O)(=O)CC1')
        print(Chem.MolToSmiles(m))
        res = RecapDecompose(m, onlyUseReactions=[2])
        children2 = res.GetLeaves().keys()
        print('Sulfonamide环磺酰胺subnodes:{}'.format(children2))
        allchildren2 = res.GetAllChildren().keys()
        print('Sulfonamide环磺酰胺all subnodes:{}'.format(allchildren2))
        Leaves2 = res.GetLeaves().keys()
        print('Sulfonamide环磺酰胺Leaf nodes:{}'.format(Leaves2))
        get_recap_tree(m, [2])

    def testAmineRxn(self):
        #Reaction 4 breaks all three bonds connected to the nitrogen atom
        # , while Reaction 5 breaks the cyclic nitrogen-bonded bond, retaining the cyclic nitrogen.
        smiles_list = [
            'C1CC1N(C1NC1)C1OC1',
            'c1ccccc1N(C1NC1)C1OC1'
            ,'c1ccccc1N(c1ncccc1)C1OC1'
            ,'c1ccccc1N(c1ncccc1)c1ccco1'
            ,'C1CCCCN1C1CC1'
            ,'C1CCC2N1CC2'
                       ]
        reactant_smarts = '[N;!$(N-C=[#7,#8,#15,#16]):1](-!@[*:2])(-!@[*:3])-!@[*:4]'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [3])
            print('Amine({})subnodes:{}'.format(smile, children))
            print('Amine({})all subnodes:{}'.format(smile, allchildren))
            print('Amine({})Leaf nodes:{}'.format(smile, Leaves))

    def testUreaRxn(self):
        smiles_list = ['C1CC1NC(=O)NC1OC1'
            ,'C1CC1NC(=O)N(C)C1OC1'
            # ,'C1CCNC(=O)NC1C'
            # ,'c1cccn1C(=O)NC1OC1'
            # ,'c1cccn1C(=O)n1c(C)ccc1'
            ,'CN(C(=O)NC1=CC=CC([N+](=O)[O-])=C1)C1=CC=CC=C1C(=O)NCC1=CC=CO1'
            ,'COC1=CC=C(CCNC(=O)C2=CC=CC=C2NC(=O)NC2=C(F)C=CC(F)=C2OC)C=C1'
                       ]
        reactant_smarts = '[#7;+0;D2:1]!@C(!@=O)!@[#7;+0;D2,D3:2]'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children,allchildren,Leaves = testfunction(smile,[4])
            print('Urea({})subnodes:{}'.format(smile,children))
            print('Urea({})all subnodes:{}'.format(smile,allchildren))
            print('Urea({})Leaf nodes:{}'.format(smile,Leaves))

    def testEtherRxn(self):
        smiles_list = ['C1CC1OC1OC1'
            ,'C1CCCCO1'
            ,'c1ccccc1OC1OC1'
            ,'c1ccccc1Oc1ncccc1'
            ]
        for smile in smiles_list:
            children,allchildren,Leaves = testfunction(smile,[5])
            print('Ether({})subnodes:{}'.format(smile,children))
            print('Ether({})all subnodes:{}'.format(smile,allchildren))
            print('Ether({})Leaf nodes:{}'.format(smile,Leaves))

    def testWittingRxn(self):
        smiles_list = ['ClC=CBr'
            ,'C1CC=CC1']
        for smile in smiles_list:
            children, allchildren, Leaves = testfunction(smile, [6])
            print('Witting({})subnodes:{}'.format(smile, children))
            print('Witting({})all subnodes:{}'.format(smile, allchildren))
            print('Witting({})Leaf nodes:{}'.format(smile, Leaves))

    def testAromNAliphCRxn(self):
        smiles_list = ['c1cccn1CCCC'
            , 'c1ccc2n1CCCC2'
                       ]
        for smile in smiles_list:
            children, allchildren, Leaves = testfunction(smile, [7])
            print('AromNAliphC({})subnodes:{}'.format(smile, children))
            print('AromNAliphC({})all subnodes:{}'.format(smile, allchildren))
            print('AromNAliphC({})Leaf nodes:{}'.format(smile, Leaves))

    def testLactamNAliphCRxn(self):
        smiles_list = ['C1CC(=O)N1CCCC'
            , 'O=C1CC2N1CCCC2'
                       ]
        for smile in smiles_list:
            children, allchildren, Leaves = testfunction(smile, [8])
            print('LactamNAliphC({})subnodes:{}'.format(smile, children))
            print('LactamNAliphC({})all subnodes:{}'.format(smile, allchildren))
            print('LactamNAliphC({})Leaf nodes:{}'.format(smile, Leaves))

    def testAromCAromCRxn(self):
        smiles_list = ['c1ccccc1c1ncccc1'
            , 'c1ccccc1C1CC1'
                       ]
        for smile in smiles_list:
            children, allchildren, Leaves = testfunction(smile, [9])
            print('AromCAromC({})subnodes:{}'.format(smile, children))
            print('AromCAromC({})all subnodes:{}'.format(smile, allchildren))
            print('AromCAromC({})Leaf nodes:{}'.format(smile, Leaves))

    def testAromNAromCRxn(self):
        '''
        Buchwald-Hartwig偶联反应
        :return:
        '''
        smiles_list = ['COC1=CC(Cl)=CC=C1C1=NC(CC2=CC=C(F)C=C2)=NO1'
            , 'COC1CCN(C2=NC3=CC=CC=C3N=C2NS(=O)(=O)C2=CC=CC=C2)CC1'
            ,'C=CCOC1=CC=C(C=O)C=C1N(CC)CC1=CC(C)=C(OCC2=CC=CC=C2C)C(C)=C1'
            ,'CN1CCN(C2=CC=C(NC3=CC4N(C)C=CC4C=N3)C=C2)CC1'
            ,'CN1CCN(C2=CC=C(NC3=NC4N(C)C=NC4C=N3)C=C2)CC1'
            ,'CN1CCN(C2=CC=C(NC3=CC4N(C)C=NC4C=N3)C=C2)CC1'
                       ]
        for smile in smiles_list:
            children, allchildren, Leaves = testfunction(smile, [10])
            print('AromNAromC({})subnodes:{}'.format(smile, children))
            print('AromNAromC({})all subnodes:{}'.format(smile, allchildren))
            print('AromNAromC({})Leaf nodes:{}'.format(smile, Leaves))

    def testTriazoleRxn(self):
        '''
        1,2,4-triazole_carboxylic-acid_ester 1,2,4-三氮唑合成（酸酯）
        :return:
        '''
        smiles_list = ['COC1=CC(Cl)=CC=C1C1=NC(C(C)(C)NC2CC2)=NN1'
            , 'COC1=CC(C(=O)CC2=NNC(C(C)C3=CC=C(OC(F)(F)F)C=C3)=N2)=CC=C1'
            ,'FC(F)(F)COC1=CC=C(C2=NNC(C(F)(F)C3CCCCC3)=N2)C=C1'
            ]
        reactant_smarts = '[c:1]1:[n:2]:[nH]:[c:3]:[n]@1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [11])
            print('Triazole({})subnodes:{}'.format(smile, children))
            print('Triazole({})all subnodes:{}'.format(smile, allchildren))
            print('Triazole({})Leaf nodes:{}'.format(smile, Leaves))

    def testOxadiazoleRxn(self):
        '''
        Oxadiazole环状N连的键打碎
        :return:
        '''
        smiles_list = ['COC1=CC(Cl)=CC=C1C1=NC(CC2=CC=C(F)C=C2)=NO1'
            , 'O=CC1=CC=C(C2=NOC(CN3C=CC4=CC=CN=C43)=N2)C=C1'
            , 'CC1=NNC(SCC2=NC(CC(=O)C3=CC=CC=C3)=NO2)=C1[N+](=O)[O-]'
                       ]
        for smile in smiles_list:
            children, allchildren, Leaves = testfunction(smile, [12])
            print('Oxadiazole({})subnodes:{}'.format(smile, children))
            print('Oxadiazole({})all subnodes:{}'.format(smile, allchildren))
            print('Oxadiazole({})Leaf nodes:{}'.format(smile, Leaves))

    def testHuisgen_CuRxn(self):
        '''
        Huisgen_Cu-catalyzed_1,4-subst
        :return:
        '''
        smiles_list = ['O=C(N1CCN(CC1)S(=O)(C2=CC=CC=C2)=O)CN3C=C(N=N3)C4CCC(CC4)CO'
            , 'C1(C2=CC=C3N=CSC3=C2)=CN(CC4=CC=C(C=C4)OC5=CC=CN=C5)N=N1'
            , 'CN1C(NCCN2C=C(N=N2)C3=CC=C4SC=CC4=C3)=NC5=CC=CC=C51'
            ,'CC1=CC(Br)=CC=C1C2=CN(N=N2)CC3CCN(CC3)C(C(C)(C)C)=O'
                       ]
        reactant_smarts = '[c:1]1:[c:2]:n(-[C:3]):n:n@1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [13])
            print('Huisgen_Cu({})subnodes:{}'.format(smile, children))
            print('Huisgen_Cu({})all subnodes:{}'.format(smile, allchildren))
            print('Huisgen_Cu({})Leaf nodes:{}'.format(smile, Leaves))

    def testHuisgen_RuRxn(self):
        '''
        Huisgen_Ru-catalyzed_1,5_subst
        :return:
        '''
        smiles_list = ['CN1C=C(C2=CN=NN2CC(=O)N2CCN(S(=O)(=O)C3=CC=CC=C3)CC2)C=N1'
            , 'CC1=C(C2=CC=CC=C2)C2=C(N=C(CN3N=NC=C3CC(C)(O)C3=CC=CS3)NC2=O)S1'
            , 'CC(=O)C(CC1=CN=NN1CCNC1=NC2=CC=CC=C2N1C)C(C)=O'
            ,'CN1CCN(CC2=CN=NN2C(CNCC(F)(F)F)C2=CC=C(F)C=C2)CC1'
                       ]
        reactant_smarts = '[c:1]1:[c:2]:n:n:n(-[C:3])@1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [14])
            print('Huisgen_Ru({})subnodes:{}'.format(smile, children))
            print('Huisgen_Ru({})all subnodes:{}'.format(smile, allchildren))
            print('Huisgen_Ru({})Leaf nodes:{}'.format(smile, Leaves))

    def testBenzimidazole1Rxn(self):
        '''
        Benzimidazole_derivatives_aldehyde  苯并咪唑合成（醛）
        :return:
        '''
        smiles_list = ['CCN1C(CN2NC(=O)C3=C(C=CC=C3)C2=O)=NC2=C1C=C(C)C=C2']
        reactant_smarts = '[c:3]12:[c:1](:[c:10]:[c:9]:[c:8]:[c:7]:1)[n:2]:[c:5](-[#6:6]):[n:4]@2'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [15])
            print('Benzimidazole1({})subnodes:{}'.format(smile, children))
            print('Benzimidazole1({})all subnodes:{}'.format(smile, allchildren))
            print('Benzimidazole1({})Leaf nodes:{}'.format(smile, Leaves))

    def testBenzoxazoleRxn(self):
        '''
        Benzoxazole_carboxylic-acid  苯并噁唑合成（酸）
        这里把[c:5](-[#6:6]):[n:4]中的冒号改成双键仍然能断但是子结构匹配会为False
        :return:
        '''
        smiles_list = ['COC1=CC(Cl)=CC=C1C1=NC2=C(O1)C([N+](=O)[O-])=CC(Cl)=C2'
            , 'CSC1CCCC1C1=NC2=C(C=CC(C(C)(C)C)=C2)O1'
            ,'CC1=CC=CC2=C1OC(C=CC1CCCN1C)=N2'
                       ]
        reactant_smarts = '[c:3]12:[c:1](:[c:10]:[c:9]:[c:8]:[c:7]:1):[o:2]:[c:5](-[#6:6]):[n:4]@2'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [16])
            print('Benzoxazole({})subnodes:{}'.format(smile, children))
            print('Benzoxazole({})all subnodes:{}'.format(smile, allchildren))
            print('Benzoxazole({})Leaf nodes:{}'.format(smile, Leaves))

    def testPictetSpenglerRxn(self):
        '''
        Pictet-Spengler  (四氢异喹啉合成) β-芳基乙胺和羰基化合物酸性条件下环化缩合得到四氢异喹啉的反应
        :return:
        '''
        smiles_list = ['CCOC1=C(OC(C)C(=O)N(C)C)C=CC(C2NCCC3=C2C=C(OC(F)(F)F)C=C3)=C1'
            , 'COC1=CC(C)=CC2=C1C(C1=CN(C)N=C1C)NCC2'
            , 'CSC1=CC=CC2=C1CCNC2C1=CC=C(OCC2=CN3C=CC=CC3=N2)C=C1'
            ,'C1=CC=C(COC2=CC=CC=C2C2NCCC3=C2C=CC2=C3NN=C2)N=C1'
                       ]
        reactant_smarts = '[c:1]12:[c:2](-[CH2:7]-[CH2:8]-[NH1:9]-[C:10]-2(-[#6:11])):[c:3]:[c:4]:[c:5]:[c:6]:1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [17])
            print('PictetSpengler({})subnodes:{}'.format(smile, children))
            print('PictetSpengler({})all subnodes:{}'.format(smile, allchildren))
            print('PictetSpengler({})Leaf nodes:{}'.format(smile, Leaves))

    def testFischerindoleRxn(self):
        '''
        Fischer indole  人名反应（合成吲哚）
        :return:
        '''
        smiles_list = ['CCN1C(C2=CC=CC=C2)=C(C2=C(C#N)C3=C(N2)C(C)=CC(C)=C3)C2=C1C=CC=C2'
            , 'CCOC(=O)C1C2=C(C3=C(N2)C(Cl)=CC([N+](=O)[O-])=C3)C1(C)C'
            , 'C=CC1=C(C)C2=C(N1)C(F)=C(I)C=C2'
                       ]
        reactant_smarts = '[c:5]2:[n:1]:[c:2]:[c:3]:[c:4]@2'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [18])
            print('Fischerindole({})subnodes:{}'.format(smile, children))
            print('Fischerindole({})all subnodes:{}'.format(smile, allchildren))
            print('Fischerindole({})Leaf nodes:{}'.format(smile, Leaves))

    def testBenzofuranRxn(self):
        '''
        Benzofuran  苯并呋喃合成
        :return:
        '''
        smiles_list = ['CC(C)(C)C1=CC(I)=CC2=C1OC(CNC(=O)C1=CC=CN=C1Cl)=C2'
            , 'CC1=CC2=C(OC(C(C)(C)CC3CCC(F)(F)CC3)=C2)C(N)=C1'
            , 'COC1=CC(CO)=CC2=C1OC(C1=CC(Br)=CC=C1N)=C2'
                       ]
        reactant_smarts = '[c:1]12:[c:2](:[c:9]:[c:8]:[c:7]:[c:6]:2):[o:3]:[c:4]:[cH:5]@1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [19])
            print('Benzofuran({})subnodes:{}'.format(smile, children))
            print('Benzofuran({})all subnodes:{}'.format(smile, allchildren))
            print('Benzofuran({})Leaf nodes:{}'.format(smile, Leaves))

    def testNiementowskiRxn(self):
        '''
        Niementowski_quinazoline  邻氨基苯甲酸与酰胺反应生成喹唑啉
        :return:
        '''
        smiles_list = ['CCOC(=O)C=C1SCC(=O)N1CC1=NC2=C(C=C(C)C=C2)C(=O)N1C1=CC(C)=CC=C1C'
            , 'COC1=CC2=C(C=C1OC)C(=O)NC(C1=CC=C(C)N=C1)=N2'
            , 'O=CC1=CC=C(C2=NC3=C(C=CC(OC(F)F)=C3)C(=O)N2)C=C1'
                       ]
        reactant_smarts = '[c:1]12:[c:2](:[c:11]:[c:10]:[c:9]:[c:8]:2):[c:3](=[O:4]):[#7:5]:[c:6]:[#7:7]@1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [20])
            print('Niementowski({})subnodes:{}'.format(smile, children))
            print('Niementowski({})all subnodes:{}'.format(smile, allchildren))
            print('Niementowski({})Leaf nodes:{}'.format(smile, Leaves))

    def testBenzothiazoleRxn(self):
        '''
        Benzothiazole  苯并噻唑合成（醛）
        :return:
        '''
        smiles_list = ['CC1=CC=CC2=C1SC(C1=CN(C3=CC=CC=C3)N=C1C1=CC=C3OCOC3=C1)=N2'
                       ]
        # reactant_smarts = '[c:3]12:[c:1](:[c:10]:[c:9]:[c:8]:[c:7]:1):[s:2]:[c:5](-[#6:6]):[n:4]:2'
        reactant_smarts = '[c:1]12:[c:2](:[c:10]:[c:9]:[c:8]:[c:7]:2):[s:3]:[c:4]([#6:6]):[n:5]:1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [21])
            print('Benzothiazole({})subnodes:{}'.format(smile, children))
            print('Benzothiazole({})all subnodes:{}'.format(smile, allchildren))
            print('Benzothiazole({})Leaf nodes:{}'.format(smile, Leaves))

    def testThiazoleRxn(self):
        '''
        Thiazole  噻唑合成
        :return:
        '''
        smiles_list = ['CCOC(=O)CC1=C(C)SC(NC2=C(C)C=CC(C)=C2)=N1'
            , 'CC1=C(C2=CC=C(OC(F)F)C=C2)N=C(NC2=CC=C3CCCCC3=C2)S1'
            , 'O=C(NC1=CC=CC=C1)C1=C(C2=CC=CC=C2)N=C(CC2=CC=CC=C2F)S1'
                       ]
        reactant_smarts = '[c:1]2(-[#6:6]):[n:2]:[c:3]:[s:4][c:5]([#6:7]):2'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [22])
            print('Thiazole({})subnodes:{}'.format(smile, children))
            print('Thiazole({})all subnodes:{}'.format(smile, allchildren))
            print('Thiazole({})Leaf nodes:{}'.format(smile, Leaves))

    def testFriedlaenderquinolineRxn(self):
        '''
        Friedlaender_quinoline  人名反应（合成喹啉）
        :return:
        '''
        smiles_list = ['CCN1C(C2=CC=CC=C2)=C(C2=NC3=C(C=C2C#N)C=C(Cl)C(Br)=C3)C2=C1C=CC=C2'
            , 'CC1=CC=C(Cl)C2=C1N=C1C(=C2)C2(CCCCC2)OC2=C1C=CC(O)=C2O'
            , 'CCOC1CC2=NC3=C(C=C21)C=CC=C3C'
                       ]
        # reactant_smarts = '[c:2]2:[c:3]:[c:4]:[c:5]:[c:6]:[n:1]:2'
        reactant_smarts = '[c:1]12:[c:2](:[n:7]:[c:8]:[c:9]:[c:10]:2):[c:3]:[c:4]:[c:5]:[c:6]:1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [23])
            print('Friedlaenderquinoline({})subnodes:{}'.format(smile, children))
            print('Friedlaenderquinoline({})all subnodes:{}'.format(smile, allchildren))
            print('Friedlaenderquinoline({})Leaf nodes:{}'.format(smile, Leaves))

    def testPhthalazinoneRxn(self):
        '''
        Phthalazinone  二氮杂萘酮合成
        :return:
        '''
        smiles_list = ['CCOCCCN1C(N2N=CC3=C(C2=O)C(F)=CC=C3F)=NC2=C(C=CC=C2)C1=O'
            , 'CC1=NN(C2=CC(=O)NC(=O)N2)C(=O)C2=C1N=CC=C2'
            , 'O=C1C2=C(C=NN1C1=CC=C(F)C=C1F)C=CC=N2'
                       ]
        reactant_smarts = '[c:1]2:[c:2]:[c:3]:[n:4]:[n:5]:[c:6](=[O:7])@2'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [24])
            print('Phthalazinone({})subnodes:{}'.format(smile, children))
            print('Phthalazinone({})all subnodes:{}'.format(smile, allchildren))
            print('Phthalazinone({})Leaf nodes:{}'.format(smile, Leaves))

    def testBenzothiopheneRxn(self):
        '''
        Benzothiophene  苯并噻吩合成
        :return:
        '''
        smiles_list = ['CC(=O)C1=CC2=C(C=C1)SC(CNC(=O)C1=CC=CN=C1Cl)=C2'
            , 'CC(=O)C1=CC2=C(C=C1)SC(CN1CCOCC1)=C2'
            , 'COC1=CC2=C(C=C(CC3COC3)S2)C=C1'
            , 'CC(=O)C1=CC2=C(C=C1)SC(C1=CC=CC(NC(C)C)=C1)=C2'
            ,'CC(=O)C1=CC2=C(C=C1)SC(CN1N=C(C)N=C1C)=C2'
                       ]
        reactant_smarts = '[c:1]2:[c:2]:[s:3]:[c:4]:[cH:5]@2'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [25])
            print('Benzothiophene({})subnodes:{}'.format(smile, children))
            print('Benzothiophene({})all subnodes:{}'.format(smile, allchildren))
            print('Benzothiophene({})Leaf nodes:{}'.format(smile, Leaves))

    def testSpirochromanoneRxn(self):
        '''
        Spirochromanone  螺环香兰酮合成
        :return:
        '''
        smiles_list = ['CC(=O)NC1CCC2(CC1)CC(=O)C1=C(O2)C(Br)=CC(C)=C1'
            , 'O=C1CC2(CCC3(CCCC3)CC2)OC2=C1C=CC(Cl)=C2Cl'
            , 'O=C1CC2(CCN(CCO)CC2)OC2=C1C=CC([N+](=O)[O-])=C2'
                       ]
        reactant_smarts = '[O:6]1-[c:5]:[c:1]-[C:2](=[OD1:3])-[C:4]-[C:7]-1'
        patt = Chem.MolFromSmarts(reactant_smarts)
        print(Chem.MolToSmiles(patt))
        for smile in smiles_list:
            m = Chem.MolFromSmiles(smile)
            print(m.HasSubstructMatch(patt))
            children, allchildren, Leaves = testfunction(smile, [26])
            print('Spirochromanone({})subnodes:{}'.format(smile, children))
            print('Spirochromanone({})all subnodes:{}'.format(smile, allchildren))
            print('Spirochromanone({})Leaf nodes:{}'.format(smile, Leaves))

if __name__ == '__main__':

    unittest.main()
