from tqdm import tqdm
from exps2 import regex, terminal, star, union, concat
from typing import Generic, TypeVar, Callable, Iterator
from dataclasses import dataclass
from itertools import count
from functools import partial, reduce

T = TypeVar('T')

@dataclass(frozen=True)
class dfa(Generic[T]):
    alphabet: set[str]
    start: T
    accepts: Callable[[T], bool]
    delta: Callable[[T, str], T]

    def __hash__(self): return hash(str(self))

    def __str__(self) -> str:
        fa = self.cannonize()
        def helper(states, seen, edge_strings, accept_strings):
            if not len(states): return edge_strings, accept_strings
            else:
                state, *others = states
                def add_edge_string(acc, sym):
                    states, seen, edge_strings = acc
                    to_state = fa.delta(state, sym)
                    edge = f'  {state} -> {to_state} [label="{sym}"]\n'
                    string = edge_strings + edge
                    if to_state not in seen: 
                        return {to_state} | states, {to_state} | seen, string
                    else: return states, seen, string
                acc = set(others), seen, edge_strings
                acc = reduce(add_edge_string, fa.alphabet, acc)
                if fa.accepts(state): a = f'{accept_strings} {state}'
                else: a = accept_strings
                return helper(*acc, a)
        edge_strings, accept_strings = helper({fa.start}, {fa.start}, "", "")
        return ('digraph {\n'
                f'  rankdir=LR\n'  
                f'  node [shape=doublecircle]{accept_strings}\n'
                f'  node [shape=point] q \n'
                f'  node [shape=circle] \n'
                f'  q -> {fa.start}\n' 
                f'{edge_strings}'
                '}')

    def __eq__(self, o: 'dfa') -> bool: return str(self) == str(o)

    def size(self) -> int:
        return sum(self.count_strs(length) for length in range(15))

    def count_strs(self, length: int, start=None, str_start=True) -> int:
        start = {self.start: 1} if start is None else start
        states: dict = dict()
        for state, multiplicity in start.items():
            for sym in self.alphabet:
                to_state = self.delta(state, sym)
                states[to_state] = states.get(to_state, 0) + multiplicity
        if not length and str_start: return 0 
        elif length: return self.count_strs(length - 1, states, str_start=False) 
        else: return sum(v for k, v in start.items() if self.accepts(k))

    def get_str(self, length, idx, state=None):
        state = self.start if state is None else state
        string = ''
        for l in reversed(range(length)):
            def is_valid_path(sym): 
                start = {self.delta(state, sym): 1}
                return self.count_strs(l, start, False)
            valid_paths = list(filter(is_valid_path, self.alphabet))
            idx, sym_idx = divmod(idx, len(valid_paths))
            sym = valid_paths[sym_idx]
            state = self.delta(state, sym)
            string += sym
        return string

    def concat(self, other: 'dfa') -> 'dfa':
        start = (self.start | other.start if self.accepts(self.start) 
                else self.start)
        def delta(state, symbol):
            nexts = self.delta(state, symbol) | other.delta(state, symbol)
            return other.start | nexts if self.accepts(nexts) else nexts
        return dfa(self.alphabet | other.alphabet, start, other.accepts, delta)

    def intersection(self, other: 'dfa') -> 'dfa':
        def accepts(state): return self.accepts(state) and other.accepts(state)
        def delta(*args): return self.delta(*args) | other.delta(*args)
        return dfa(
                self.alphabet | other.alphabet,
                self.start | other.start,
                accepts, delta)

    def union(self, other: 'dfa') -> 'dfa':
        def accepts(state): return self.accepts(state) or other.accepts(state)
        def delta(*args): return self.delta(*args) | other.delta(*args)
        return dfa(
                self.alphabet | other.alphabet,
                self.start | other.start,
                accepts, delta)

    def complement(self):
        def accepts(state): return not self.accepts(state)
        return dfa(self.alphabet, self.start, accepts, self.delta)

    def get_ambiguity(self, transitions: str, verbose=False):
        ambiguity = 0
        state = self.start
        for transition in transitions:
            if verbose: print(state)
            state = self.delta(state, transitions)
            ambiguity = max(ambiguity, len(state))
        if verbose: print(state)
        return ambiguity

    def get_mental_load(self, transitions: str):
        d = self.cannonize()
        state = d.start
        states = [state]
        for transition in transitions:
            state = d.delta(state, transition)
            states.append(state)
        return len(set(states))

    def cannonize(self):
        fa = self.minimize()
        def add_state(state, assoc, idx):
            def add_edge(acc, sym):
                assoc, idx = acc
                trans = fa.delta(state, sym)
                if trans in assoc: return acc
                else: return add_state(trans, {trans: idx} | assoc, idx + 1)
            return reduce(add_edge, sorted(fa.alphabet), (assoc, idx))
        assoc, _ = add_state(fa.start, {fa.start: 0}, 1)
        inv_assoc = dict(zip(assoc.values(), assoc.keys()))
        def accepts(state): return fa.accepts(inv_assoc[state])
        def delta(state, symbol): return assoc[fa.delta(inv_assoc[state], symbol)]
        return dfa(fa.alphabet, 0, accepts, delta)

    def minimize(self):
        all_states = self.get_states()
        def get_states_into(states: frozenset[T], sym: str) -> frozenset[T]:
            def into(state): return self.delta(state, sym) in states
            return frozenset(filter(into, all_states))
        def f(s1, s2): return len(s1 & s2) and len(s2 - s1)
        def partition(p, w) -> set[frozenset[T]]:
            if not len(w): return p
            else: 
                hd, *tl = w
                def refine_on(acc, sym):
                    s1 = get_states_into(hd, sym)
                    def repartition(acc, s2):
                        p, w = acc
                        i, d = s1 & s2, s2 - s1
                        p = {d, i} | p - {s2}
                        if s2 in w: w = {d, i} | w - {s2}
                        elif len(i) > len(d): w = {d} | w
                        else: w = {i} | w
                        return p, w
                    choices = filter(partial(f, s1), acc[0])
                    return reduce(repartition, choices, acc)
                return partition(*reduce(refine_on,self.alphabet, (p, set(tl))))
        accept_states = frozenset(filter(self.accepts, all_states))
        non_accepts = frozenset(all_states - accept_states)
        p = {accept_states, non_accepts}
        hopcroft: set[frozenset[T]] = partition(p, p)
        def find(q): return next(state for state in hopcroft if q in state)
        def accepts(state): return any(self.accepts(s) for s in state)
        def delta(state, symbol): 
            s, *_ = state
            return find(self.delta(s, symbol))
        return dfa(self.alphabet, find(self.start), accepts, delta)

    def get_states(self) -> frozenset[T]:
        def add_state(state, states): 
            def explore(states, sym):
                next_state = self.delta(state, sym)
                if next_state in states: return states
                else: return add_state(next_state, {next_state} | states)
            return reduce(explore, self.alphabet, states)
        return add_state(self.start, {self.start})

# Make available in module
def complement(fa): return fa.complement()
def get_states(fa): return fa.get_states()
def minimize(fa): return fa.minimize()
def cannonize(fa): return fa.cannonize()

def dfa_of_regex(
        a: set[str], 
        e: regex, 
        nats: Iterator[int] = count(0)) -> dfa[frozenset[int]]:
    def isin(q: int, s: set[int]) -> bool: return q in s
    q, f = next(nats), next(nats)
    match e:
        case terminal(t): 
            def d2(state, symbol):
                if q in state and symbol == t: return frozenset({f})
                else: return frozenset()
            return dfa(a, frozenset({q}), partial(isin, f), d2)
        case star(t): 
            inner = dfa_of_regex(a, t)
            start = inner.start | {f}
            def d3(state, symbol): 
                inner_next = inner.delta(state, symbol)
                if inner.accepts(inner_next): return start | inner_next
                else: return inner_next
            return dfa(inner.alphabet, start, partial(isin, f), d3)
        case union(_):
            fa1, fa2 = map(partial(dfa_of_regex, a), e)
            return fa1.union(fa2)
        case concat(_):
            f1, f2 = map(partial(dfa_of_regex, a), e)
            return f1.concat(f2)
    print(e)
    raise TypeError

emptylanguage = dfa(
        set(), frozenset(), lambda x: False, 
        lambda state, symbol: frozenset())

emptystring = dfa(set(), frozenset({-1}), lambda x: -1 in x, 
        lambda state, symbol: frozenset())

def dot_depth(fa: dfa, alphabet: set[str] = set('ab')) -> int:
    """ Does not halt. """
    singletons = frozenset(dfa_of_regex(set('ab'), terminal(t)) for t in alphabet) 
    E = singletons | {emptylanguage, emptystring} 
    def M(X): return closure(X, (concat,))
    def B(X): return closure(X, (union, intersection), (complement,))
    def closure(X, binops, unops=()):
        def apply_ops(X):
            return (X.union(op(x, x_) for x in X for x_ in X for op in binops)
                    .union(unop(x) for x in X for up in unops))
        X_ = apply_ops(X)
        while X_ != X: 
            print(len(X))
            X = X_
            X_ = apply_ops(X)
        return X
    depth = 0
    dot_depth = B(M(E))
    while fa not in dot_depth: 
        depth += 1
        dot_depth = B(M(dot_depth))
        print(len(dot_depth))
    return depth

if __name__ == '__main__':
    re = concat(
            star(terminal('a')), 
            concat(
                terminal('b'), 
                star(union(terminal('a'), terminal('b')))))
    fa = dfa_of_regex(set('ab'), re)
    ca = complement(fa)
    print(fa)


