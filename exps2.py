from functools import cache, partial
from typing import NamedTuple, Union  

class terminal(NamedTuple): 
    s: str
    def __str__(self): return self.s
    def __repr__(self): return str(self)
    def pleb(self): return f'%||%</{self.s}>'
    
class star(NamedTuple): 
    e: Union[terminal, 'union', 'concat']
    def __str__(self): 
        match self.e:
            case union(_) | concat(_): return f'({str(self.e)})*'
            case _: return f'{str(self.e)}*'
    def __repr__(self): return str(self)
    def pleb(self): return f'* {self.e.pleb()}'
    @staticmethod
    @cache
    def count(a, d): return a if d == 1 else count_binary(a, d - 1) * 2
    @staticmethod
    def get(alphabet, d, i):
        union_idxs = union.count(len(alphabet), d - 1)
        if d == 1: return star(terminal(alphabet[i]))
        elif i < union_idxs: return star(union.get(alphabet, d - 1, i))
        else: return star(concat.get(alphabet, d - 1, i - union_idxs))

class union(NamedTuple): 
    e1: Union[terminal, star, 'concat']
    e2: 'regex'
    def __str__(self): return f'{self.e1}|{self.e2}'
    def __repr__(self): return str(self)
    def pleb(self): return f'\/ ({self.e1.pleb()}, {self.e2.pleb()})'
    def __eq__(self, re: 'regex') -> bool:
        return type(re) == union and tuple(self) == tuple(re)
    @staticmethod
    def count(a, d): return count_binary(a, d)
    @staticmethod
    def get(alphabet, d, idx): 
        return get_binary(alphabet, union, concat, d, idx)

class concat(NamedTuple):
    e1: Union[terminal, star, union]
    e2: 'regex'
    def __repr__(self): return str(self)
    def pleb(self): return f'@ ({self.e1.pleb()}, {self.e2.pleb()})'
    def __eq__(self, re: 'regex') -> bool:
        return type(re) == concat and tuple(self) == tuple(re)
    def __str__(self): 
        def paren(e):
            match e:
                case union(_): return f'({str(e)})'
                case _: return str(e)
        return f'{paren(self.e1)}{paren(self.e2)}'
    @staticmethod
    def count(a, d): return count_binary(a, d)
    @staticmethod
    def get(alphabet, d, idx): 
        return get_binary(alphabet, concat, union, d, idx)

regex = terminal | star | union | concat

def count_exps(a: int, d: int) -> int: 
    if not d: return a
    else: return star.count(a, d) + union.count(a, d) + concat.count(a, d)

@cache
def count_subexps(a: int, d: int, i: int) -> int:
    e1 = count_exps(a, i) - count_binary(a, i) 
    e2 = count_exps(a, d - i - 1)
    return e1 * e2

@cache
def count_binary(a: int, d: int) -> int:
    return 0 if not d else sum(count_subexps(a, d, i) for i in range(d))

count_structs = partial(count_exps, 1)

def get_regex(alphabet: str, d: int, i: int) -> regex:
    assert d >= 0
    assert i < count_exps(len(alphabet), d)
    star_idxs = star.count(len(alphabet), d)
    union_idxs = star_idxs + union.count(len(alphabet), d)
    if not d: return terminal(alphabet[i])
    elif i < star_idxs: return star.get(alphabet, d, i)
    elif i < union_idxs: return union.get(alphabet, d, i - star_idxs)
    else: return concat.get(alphabet, d, i - union_idxs)

def get_binary(alphabet, op, other, d, idx): 
    assert d > 0
    idxs = 0
    for b in range(d):
        subexps = count_subexps(len(alphabet), d, b)
        idxs += subexps
        if idx < idxs: 
            i = idx - idxs + subexps
            p, q = divmod(i, count_exps(len(alphabet), d - b - 1))
            star_idxs = star.count(len(alphabet), b)          
            tl = get_regex(alphabet, d - b - 1, q)
            if not b: return op(terminal(alphabet[p]), tl)
            elif p < star_idxs: return op(star.get(alphabet, b, p), tl)
            else: return op(other.get(alphabet, b, p - star_idxs), tl)
    raise ValueError('idx too large')

def get_terminals(re: regex) -> set[str]: 
    match re:
        case terminal(s): return set(s)
        case star(_) | union(_) | concat(_): 
            return set.union(set(), *map(get_terminals, re)) 
    raise TypeError

def get_constituents(re: regex) -> set[regex]:
    def flatten_binop(binop, exp: regex) -> tuple[regex, ...]:
        match exp: 
            case binop(l, r): return l, *flatten_binop(binop, r)
            case _: return exp,
    def get_subsequences(seq: tuple[regex, ...]) -> list[tuple[regex, ...]]:
        out = list()
        for l in range(2,len(seq)+1):
            for start in range(len(seq)-l+1):
                out.append(seq[start:start+l])
        return out
    def make_binop(binop, seq: tuple[regex,...]) -> regex:
        match seq:
            case last,: return last
            case hd, *tl: return binop(hd, make_binop(binop, tl))
        raise TypeError
    match re:
        case terminal(_): return {re}
        case star(e): return {re, *get_constituents(e)}
        case union(_) | concat(_): 
            subexps = flatten_binop(type(re), re)
            subseqs = get_subsequences(subexps)
            binops = set(map(partial(make_binop, type(re)), subseqs))
            atoms = map(get_constituents, subexps)
            return binops.union(*atoms)
    raise TypeError

def get_structure(re: regex) -> tuple:
    match re: 
        case terminal(_): return ()
        case star(e): return get_structure(e),
        case union(e1, e2): 
            match e2:
                case union(_): 
                    return (get_structure(e1), *get_structure(e2))
                case _: 
                    return (get_structure(e1), get_structure(e2))
        case concat(e1, e2): 
            match e2:
                case concat(_): 
                    return (get_structure(e1), *get_structure(e2))
                case _: return (get_structure(e1), get_structure(e2))
    print(re)
    raise TypeError

def graph_of_regex(re: regex) -> set[tuple[str, str]]:
    match re:
        case terminal(_): return set()
        case star(e): 
            return {(str(type(re)), str(type(e))), *graph_of_regex(e)}
        case union(e1, e2) | concat(e1, e2): 
            tail = e2
            children: list[regex] = [e1]
            while type(tail) == type(re):
                children.append(tail[0])
                tail = tail[1]
            children.append(tail)
            return ({(str(type(re)), str(type(c))) for c in children}
                    .union(*map(graph_of_regex, children))
                    .union({(str(type(c1)), str(type(c2))) 
                        for i, c1 in enumerate(children)
                        for j, c2 in enumerate(children) if i != j}))
    raise TypeError

def n_LS(graph: set[tuple[str, str]]) -> set[frozenset[tuple[str, str]]]:
    ...

def get_reverse(re: regex):
    match re: 
        case terminal(_): return re
        case star(e): return star(get_reverse(e))
        case union(e1, e2): return union(get_reverse(e2), get_reverse(e1))
        case concat(e1, e2): return concat(get_reverse(e2), get_reverse(e1))
    raise TypeError

def get_substructures(structure: tuple) -> set[tuple]:
    return {structure}.union(*map(get_substructures, structure))

def get_set_structures(re_set: frozenset[regex]) -> set[tuple]:
    return set.union(*map(get_substructures, map(get_structure, re_set)))

def depth_of_regex(re: regex):
    match re:
        case terminal(_): return 0
        case star(_) | union(_) | concat(_): 
            return 1 + sum(map(depth_of_regex, re))
    raise TypeError

def star_height(re: regex):
    match re:
        case terminal(_): return 0
        case star(inner): return 1 + star_height(inner)
        case union(_) | concat(_): return max(map(star_height, re))
    raise TypeError

if __name__ == '__main__':
    n = 4
    res = set()
    for d in range(n):
        alphabet = 'ab'
        for j in range(count_exps(len(alphabet), d)):
            re = get_regex(alphabet, d, j)
            print(re)
            assert re not in res
            res.add(str(re))
