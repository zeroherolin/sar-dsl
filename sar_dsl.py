from frontend.pyast import Func2AST
from frontend.pyast import export_ast_json
from frontend.type_opt import DataTypeChecker
from frontend.type_opt import SubscriptModifier
from frontend.mlir_gen import MLIRGenerator


def build(func):
    ast = Func2AST(func)
    ast.preprocess(verbose=True)
    ast.parse(verbose=True)
    ast.link_parent()
    export_ast_json(ast.tree, 'ast.json', verbose=True)
    print('\n- adding node.vartype to AST node...')
    sm = SubscriptModifier(verbose=True)
    sm.visit(ast.tree)
    print('\n- checking func param type...')
    dtc = DataTypeChecker(verbose=True)
    dtc.visit(ast.tree)
    print(f"\ntotal checked params: {dtc.param_cnt}")

    mlir = MLIRGenerator(ast.tree)
    mlir.gen()
