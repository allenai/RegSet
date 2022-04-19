import functools, os, argparse, re, json, operator, math, itertools
import eval_analysis, exps2, dfa2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cache_loader import get_train_re_strs, get_lookup

tqdm.pandas()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='v8/grok/test_eval.tsv')
    parser.add_argument('-v', '--version',  default='v8')
    parser.add_argument('-t', '--train-file', 
            default='v8/train/1000_20_data/train.jsonl')
    args = parser.parse_args()
    test = get_test_set(args.filename, args.version, args.train_file)

@functools.cache
def get_balance(re: exps2.regex) -> int:
    dfa = dfa2.dfa_of_regex(set('ab'), re)
    return sum(dfa.count_strs(i) for i in range(15))

def get_ambiguity(lookup, row) -> int:
    re = lookup.get(row.instruction)
    d = dfa2.dfa_of_regex(set('ab'), re)
    return d.get_ambiguity(row.string)

def get_mental_load(lookup, row) -> int:
    re = lookup.get(row.instruction)
    dfa = dfa2.dfa_of_regex(set('ab'), re)
    return dfa.get_mental_load(row.string)

def get_test_set(filename, version, train_file):
    lookup = get_lookup(version)
    def map_on_regex(f):
        f = functools.cache(f)
        return lambda df: df.instruction.progress_map(
                lambda x: f(lookup.get(x)))
    def map_on_instructions(f): 
        f = functools.cache(f)
        return lambda df: df.instruction.progress_map(f)
    def map_mental_load(df: pd.DataFrame):
        return df.progress_apply(
                functools.partial(get_mental_load, lookup), 
                axis='columns')
    def map_ambiguity(df: pd.DataFrame):
        return df.progress_apply(
                functools.partial(get_ambiguity, lookup), 
                axis='columns')
    with open('is_starfree.json', 'r') as f:
        is_starfree = json.load(f)
    train_substructs = exps2.get_set_structures(
            set(map(lookup.get, get_train_re_strs(train_file))))
    train_constits = set.union(
            *map(exps2.get_constituents, 
                map(lookup.get, get_train_re_strs(train_file))))
    test = (eval_analysis
            .get_df(filename)
            .assign(
                ambiguity=map_ambiguity,
                strlen=lambda x: x.string.map(len),
                mental_load=map_mental_load,
                depth=map_on_regex(exps2.depth_of_regex),
                unseen_structs=map_on_regex(
                    lambda x: len(exps2.get_substructures(
                        exps2.get_structure(x)) -
                        train_substructs)),
                unseen_constits=map_on_regex(
                    lambda x: len(exps2.get_constituents(x) 
                        - train_constits)),
                balance=map_on_regex(get_balance),
                has_star=map_on_instructions(lambda x: '*' in x),
                has_union=map_on_instructions(lambda x: '|' in x), 
                has_concat=map_on_instructions(
                    lambda x: bool(re.search('(a|b)(a|b)', x))),
                structure=map_on_instructions(
                    lambda x: x.replace('b', 'a')),
                is_starfree=map_on_instructions(
                    functools.partial(
                        operator.getitem,
                        is_starfree))))
    test.to_csv(os.path.join(os.path.dirname(filename), 'attrs.csv'))
    return test

if __name__ == '__main__':
    main()

