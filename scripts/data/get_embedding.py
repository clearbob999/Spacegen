import argparse
import numpy as np
import multiprocessing as mp
from utils import get_morgen_fingerprint
import pickle


def worker(smi,args):
    return get_morgen_fingerprint(smi, nBits=args.feature)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/CDK9_matched_bbs.txt')
    # parser.add_argument('--output', type=str, default='data/matched_bbs_emb_256.npy')
    parser.add_argument('--output', type=str, default='data/CDK9_matched_bbs_emb_256.npy')
    parser.add_argument('--feature', type=int, default=256)
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    with open(args.input, 'r') as f:
        data = [l.strip().split()[0] for l in f.readlines()]
    print('the number of fragements =', len(data))

    with mp.Pool(processes=20) as pool:
        # embeddings = pool.map(worker, data)
        embeddings = pool.starmap(worker, [(smi, args) for smi in data])

    # with open(args.output, 'wb') as f:
    #     pickle.dump(embeddings, f)
    np.save(args.output, np.array(embeddings))



    

