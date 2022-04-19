import argparse
import pickle
from glob import glob
from functools import partial
from os import makedirs, path
from random import sample, seed
from typing import TypeVar, Callable
from tqdm.std import tqdm
from dfa2 import complement, dfa_of_regex, dfa
from exps2 import count_exps, get_regex, regex
from json import dump
import pandas as pd

print = partial(print, sep='\t')
def section(): print('-' * 10)

def sample_langs(dest, alphabet, depths):
    langs = set()
    exps = list()
    print('Depth', 'Exps', 'Languages')
    for d in range(depths):
        depth_d_exps = list()
        expressions = count_exps(len(alphabet), d)
        for idx in tqdm(sample(range(expressions), expressions), leave=False):
            exp = get_regex(alphabet, d, idx)
            lang = str(dfa_of_regex(set(alphabet), exp))
            if lang not in langs: 
                langs.add(lang)
                depth_d_exps.append(exp)
            else: continue
        exps.append(depth_d_exps)
        print(d, expressions, len(langs))
    with open(path.join(dest, 'cache.pickle'), 'wb') as f:
        pickle.dump(exps, f)

    section()
    df = pd.DataFrame(list((d, str(e)) for d, es in enumerate(exps) for e in es),
            columns=('depth', 'regex'))
    print(df)
    return exps
    
def main():
    seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dest', required=True)
    parser.add_argument('--depths', default=7, type=int)
    args = parser.parse_args()
    dest, depths = args.dest, args.depths
    makedirs(dest, exist_ok=True)

    train_magnitude = 4
    _, val_size, test_size = pow(10, train_magnitude), 200, 500
    alphabet = 'ab'

    section()
    if path.join(dest, 'cache.pickle') in glob(path.join(dest, '*')):
        with open(path.join(dest, 'cache.pickle'), 'rb') as f:
            exps = pickle.load(f)
    else: exps = sample_langs(dest, alphabet, depths)

    section()
    train, test = split(exps, test_size)
    train, val = split(train, val_size)
    print('Set', 'Count', 'By depth')
    print('All', len(flatten(exps)), [len(d) for d in exps])
    print('Train', len(flatten(train)), [len(d) for d in train])
    print('Val', len(flatten(val)), [len(d) for d in val])
    print('Test',len(flatten(test)), [len(d) for d in test])

    section()
    print('Train', 'Val', 'Test')
    print(len(flatten(train)), len(flatten(val)), len(flatten(test)))

    val_set = get_datasets(alphabet, val_size * 2, flatten(val))
    test_set = get_datasets(alphabet, 2 * pow(10, train_magnitude), 
            flatten(test), max_length=15, balance=True)
    test_dir = path.join(dest, 'test') 
    makedirs(test_dir, exist_ok=True)
    with open(path.join(test_dir, 'test.jsonl'), 'w') as f: 
        to_file(test_set, f, 1)
    assert val_set and test_set

    section()
    print('exps','ex per', 'total ex')
    expression_counts = [1000]
    dataset_sizes = [20]
    k = 2 * pow(10, train_magnitude)
    for exp_count in expression_counts:
        for ex_count in dataset_sizes:
            if exp_count * ex_count <= k:
                print(exp_count, ex_count, exp_count * ex_count)
                save_train_set(dest, alphabet, exp_count, ex_count, train,
                        val_set, depths, k // (exp_count * ex_count))

T = TypeVar('T')

def save_train_set(dest, alphabet, exp_count, ex_count, train, val_set,
        max_depth, k):
    def count_train_exps(depth): return len(train[depth])
    def get_exps(depth, n): return train[depth][:n]
    train_exps = flatten(
            uniform_sample(get_exps, max_depth, exp_count, count_train_exps))
    train_set = get_datasets(alphabet, ex_count * exp_count, train_exps)
    p = path.join(dest, 'train', f'{exp_count}_{ex_count}_data')
    makedirs(p, exist_ok=True)
    fname = path.join(p, 'train.jsonl')
    with open(fname, 'w') as f: to_file(train_set, f, k)
    fname = path.join(p, 'dev.jsonl')
    with open(fname, 'w') as f: to_file(val_set, f, 1)

def to_file(d, f, k):
    for _ in tqdm(range(k), leave=False):
        for re, examples in d.items():
            for e in examples['pos']:
                context = str.join(' ', (str(re), e))
                dump(dict(context=context, answer='True'), f)
                f.write('\n')
            for e in examples['neg']:
                context = str.join(' ', (str(re), e))
                dump(dict(context=context, answer='False'), f)
                f.write('\n')

def flatten(l: list[list[T]]) -> list[T]: return [i for j in l for i in j]

def split(exps, val_size):
    def exp_counter(depth): return len(exps[depth]) // 2
    def get_exps(depth, n): return exps[depth][:n]
    test = uniform_sample(get_exps, len(exps), val_size, exp_counter)
    ft = flatten(test)
    train = [[r for r in d if r not in ft] for d in exps]
    return train, test
 
metaset = dict[regex, dict[str, list[str]]]

def get_datasets(
        exps: list[regex], 
        max_length: int = 15,
        alphabet: str = 'ab', 
        target_size: int = 2 * pow(10, 2), 
        balance: bool = False) -> metaset:
    def count_strs(d):                 
        return sum(d.count_strs(l) for l in range(1, max_length))
    def get_dfas(re_idx):
        re = exps[re_idx]
        d = dfa_of_regex(set(alphabet), re)
        c = complement(d)
        return d, c
    def count_pos_neg(re_idx):
        d, c = get_dfas(re_idx)
        return count_strs(d), count_strs(c)
    examples_count = sum(pow(len(alphabet), l) for l in range(max_length)) - 1
    def count_examples(re_idx): 
        if not balance: return examples_count
        else: return 2 * min(count_pos_neg(re_idx))
    def sample_strs(d: dfa, n: int) -> list[list[str]]:
        def get_strs(length, n):
            def get_str(str_idx, _): return [d.get_str(length, str_idx)]
            return flatten(uniform_sample(get_str, d.count_strs(length), n))
        return uniform_sample(get_strs, max_length, n, d.count_strs)
    def getter(re_idx, count):
        pos_count, neg_count = count_pos_neg(re_idx)
        if pos_count <= count // 2: neg_count = count - pos_count
        elif neg_count <= count // 2: pos_count = count - neg_count
        else: 
            pos_count = count // 2
            neg_count = count - pos_count
        d, c = get_dfas(re_idx)
        pos = flatten(sample_strs(d, pos_count))
        neg = flatten(sample_strs(c, neg_count))
        assert count == len(pos) + len(neg)
        return [dict(pos=pos, neg=neg)]
    examples = flatten(uniform_sample(getter, len(exps), target_size, count_examples))
    return dict(zip(exps, examples))

def uniform_sample(
        getter: Callable[[int, int], list[T]],
        bins: int, 
        remaining: int,
        counter: Callable[[int], int] = lambda x: 1,
        leave: bool = False,
        **kwargs) -> list[list[T]]:
    a = list(filter(counter, range(bins)))
    if remaining <= len(a): 
        choices = sample(a, remaining)
        return [[] if b not in choices else getter(b, 1) for b in range(bins)]
    else:
        idxs = sorted(range(bins), key=counter)
        samples = [[] for _ in range(bins)]
        for seen, b in tqdm(list(enumerate(idxs)), leave=leave, **kwargs):
            u = remaining // (bins - seen)
            count = min(counter(b), u)
            remaining -= count
            samples[b] = getter(b, count)
        return samples

if __name__ == '__main__':
    main()
        

