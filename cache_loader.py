import pickle, jq, argparse, json
from functools import cache
from exps2 import regex

@cache
def get_lookup(version: str) -> dict[str, regex]:
    with open(f'{version}/cache.pickle', 'rb') as cachefile:
        cache = pickle.load(cachefile)
    flat = [re for depth in cache for re in depth]
    return dict(zip(map(str, flat), flat))

@cache
def get_train_re_strs(train_file: str) -> list[str]:
    with open(train_file, 'r') as f: 
        context = jq.compile('.context').input(text=f.read())
    return list(c.split(' ')[0] for c in context)

@cache
def get_starfree_re_strs(version: str) -> list[str]:
    with open(f'{version}/is_starfree.json', 'r') as f:
        return json.load(f)

@cache
def get_train_instances(train_file: str) -> list[str]:
    with open(train_file, 'r') as f:
        return list(jq.compile('.context').input(text=f.read()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', default='v8')
    args = parser.parse_args()
    lookup = get_lookup(args.version)
    train_strs = get_train_re_strs(args.version)
    print(len(lookup), 'expressions')
    print(len(train_strs), 'training expressions')

