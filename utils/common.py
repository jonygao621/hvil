import json

from .dicttree import DictTree


def json2dict(parsed_args, keys):
    d = DictTree()
    for k in keys:
        v = getattr(parsed_args, k)
        try:
            d[k] = DictTree(json.loads(v))
        except json.decoder.JSONDecodeError:
            try:
                with open(v, 'r') as f:
                    d[k] = DictTree(json.load(f))
            except FileNotFoundError:
                d[k] = DictTree(name=v)
    return d
