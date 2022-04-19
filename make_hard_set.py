from tqdm import tqdm
import pandas as pd
import pickle, json, random
from functools import partial, cache
import cache_loader 
import sample_v2
import argparse
from dfa2 import dfa_of_regex
from exps2 import regex
from typing import Callable
from compute_properties import get_balance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    exp_filters = [is_not_in_train, is_nonstarfree, is_big(64)]
    str_filters = [high_mental_load(4)]
    get_dataset(exp_filters, str_filters, debug=args.debug)

def is_in_train(exp: regex) -> bool:
    train_strs = cache_loader.get_train_re_strs('data/explore/train.jsonl')
    return str(exp) in train_strs

def is_not_in_train(exp): return not is_in_train(exp)

def is_big(i):
    def f(exp: regex) -> bool:
        return get_balance(exp) > i
    return f

def high_mental_load(i):
    def f(exp: regex, string: str) -> bool:
        return dfa_of_regex(set('ab'), exp).get_mental_load(string) > i
    return f

@cache
def get_starfree_lookup():
    with open('is_starfree.json', 'r') as f:
        return json.load(f)

def is_starfree(exp: regex) -> bool:
    return get_starfree_lookup().get(str(exp))

def is_nonstarfree(exp: regex) -> bool: 
    return not is_starfree(exp)

def str_is_not_in_train(exp: regex, string: str) -> bool:
    train_instances = cache_loader.get_train_instances('data/explore/train.jsonl')
    return ' '.join((str(exp), string)) not in train_instances

def get_dataset( 
        exp_filters: list[Callable[[regex], bool]] = list(), 
        str_filters: list[Callable[[regex, str], bool]] = list(),
        debug: bool = False) -> None:

    lookup = cache_loader.get_lookup('v8')

    def filter_func(x: regex) -> bool: 
        return all(f(x) for f in exp_filters)

    filtered_exps = list(filter(filter_func, tqdm(lookup.values())))
    print(len(filtered_exps))

    datasets = sample_v2.get_datasets(
            filtered_exps,
            target_size=2*pow(10, 6), 
            balance=True,
            max_length=3 if debug else 15,
            )

    def str_filter_func(re, string) -> bool:
        return all(f(re, string) for f in str_filters)

    dataset = dict()
    for exp, exs in tqdm(datasets.items()):
        dataset[exp] = dict()
        partial_filter = partial(str_filter_func, exp)
        for label, strs in exs.items():
            filtered_strs = list(filter(partial_filter, strs))
            dataset[exp][label] = random.sample(filtered_strs, len(filtered_strs))
        for label, strs in dataset[exp].items():
            dataset[exp][label] = strs[:min(map(len, dataset[exp].values()))]
    with open('hard.jsonl', 'w') as f: sample_v2.to_file(dataset, f, 1)

if __name__ == '__main__':
    main()
