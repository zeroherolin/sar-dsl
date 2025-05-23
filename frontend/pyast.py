import ast
import inspect
import textwrap
import astpretty
import json


class Func2AST(object):
    def __init__(self, func):
        self.func = func
        self.src = None
        self.tree = None
        self.src_np = None

    def preprocess(self, verbose=False):
        self.src, _ = inspect.getsourcelines(self.func)
        self.src = [textwrap.fill(line, tabsize=4, width=9999) for line in self.src]
        self.src = textwrap.dedent("\n".join(self.src))
        if verbose:
            print(f"source code:\n{self.src}\n")

    def parse(self, verbose=False):
        if not self.src:
            raise RuntimeError('preprocess() must be called before parse()')
        self.tree = ast.parse(self.src)
        if verbose:
            print('AST module:')
            astpretty.pprint(self.tree, indent=2, show_offsets=False)

    def link_parent(self):
        if not self.tree:
            raise RuntimeError('parse() must be called before link_parent()')
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node


def _ast_node_to_dict(node):
    if not isinstance(node, ast.AST):
        raise RuntimeError('expected AST, got %r' % node.__class__.__name__)
    result = {'node_type': node.__class__.__name__}
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            result[field] = [_ast_node_to_dict(item) for item in value]
        elif isinstance(value, ast.AST):
            result[field] = _ast_node_to_dict(value)
        else:
            result[field] = value
    return result


def get_ast_dict(node):
    if not node:
        raise RuntimeError('parse() must be called before get_ast_dict()')
    return _ast_node_to_dict(node)


def export_ast_json(node, export='ast.json', verbose=False):
    with open(export, 'w', encoding='utf-8') as f:
        json.dump(get_ast_dict(node), f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"AST has been exported to {export}")
