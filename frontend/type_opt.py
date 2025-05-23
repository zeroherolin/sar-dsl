import ast
import numpy as np


class DataType(object):
    def __init__(self, bits, dec):
        self.bits = bits
        self.dec = dec
        self.shape = None

    def __repr__(self):
        if isinstance(self.shape, int):
            raise TypeError("vectors must be of array type")
        else:
            shape_str = f"[{', '.join(map(str, self.shape))}]" if self.shape else ""
            if self.__class__.__name__ == 'Fixed':
                return f"Fixed(bits={self.bits}, dec={self.dec}){shape_str}"
            elif self.__class__.__name__ == 'UFixed':
                return f"UFixed(bits={self.bits}, dec={self.dec}){shape_str}"
            elif self.__class__.__name__ == 'Int':
                return f"Int(bits={self.bits}){shape_str}"
            elif self.__class__.__name__ == 'UInt':
                return f"UInt(bits={self.bits}){shape_str}"
            elif self.__class__.__name__ == 'Float':
                return f"Float(bits={self.bits}){shape_str}"
            elif self.__class__.__name__ == 'Complex':
                return f"Complex(bits={self.bits}){shape_str}"
            elif self.__class__.__name__ == 'Bool':
                return f"Bool(bits=1){shape_str}"
            else:
                raise NotImplementedError

    def __getitem__(self, shape):
        self.shape = shape
        return self

    def numpy_type(self):
        if self.__class__.__name__ in ('Fixed', 'Int'):
            return np.dtype(f"int{self.bits}")
        elif self.__class__.__name__ in ('UFixed', 'UInt'):
            return np.dtype(f"uint{self.bits}")
        elif self.__class__.__name__ == 'Float':
            return np.dtype(f"float{self.bits}")
        elif self.__class__.__name__ == 'Complex':
            return np.dtype(f"complex{self.bits}")
        elif self.__class__.__name__ == 'Bool':
            return np.dtype(f"bool_")
        else:
            raise NotImplementedError

    def numpy_shape(self):
        return self.shape

    def create_numpy(self, data=None):
        dtype = self.numpy_type()
        shape = self.numpy_shape()
        if data is None:
            return np.empty(shape, dtype=dtype)
        else:
            return np.asarray(data, dtype=dtype)


class Fixed(DataType):
    def __init__(self, bits, dec):
        super().__init__(bits, dec)


class UFixed(DataType):
    def __init__(self, bits, dec):
        super().__init__(bits, dec)


class Int(DataType):
    def __init__(self, bits, dec=0):
        super().__init__(bits, dec)


class UInt(DataType):
    def __init__(self, bits, dec=0):
        super().__init__(bits, dec)


class Float(DataType):
    def __init__(self, bits, dec=0):
        super().__init__(bits, dec)


class Complex(DataType):
    def __init__(self, bits, dec=0):
        super().__init__(bits, dec)


class Bool(DataType):
    def __init__(self, bits=1, dec=0):
        super().__init__(bits, dec)


type_class_mapping = {
    "Fixed": Fixed,
    "UFixed": UFixed,
    "Int": Int,
    "UInt": UInt,
    "Float": Float,
    "Complex": Complex,
    "Bool": Bool,

    "bool": Bool,
    "int32": Int,
    "uint32": UInt,
    "int64": Int,
    "uint64": UInt,
    "float32": Float,
    "float64": Float,
    "complex64": Complex,
    "complex128": Complex,
}


bool = Bool()
int32 = Int(32)
uint32 = UInt(32)
int64 = Int(64)
uint64 = UInt(64)
float32 = Float(32)
float64 = Float(64)
complex64 = Complex(64)
complex128 = Complex(128)


class _BaseVisitor(object):
    def visit(self, node):
        if node is None:
            return None
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node, config=None):
        pass

class NodeVisitor(_BaseVisitor, ast.NodeVisitor):
    pass


class NodeTransformer(_BaseVisitor, ast.NodeTransformer):
    pass


class SubscriptModifier(NodeTransformer):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def visit_Subscript(self, node):
        if self.verbose:
            print(f"\nvisiting {node.__class__.__name__} node...\n")
            print(f"parent node: {node.parent}")
        if isinstance(node.parent, ast.arg) or isinstance(node.parent, ast.AnnAssign):
            if node.parent.annotation is node:
                node.vartype = ast.unparse(node)
                if self.verbose:
                    print(f"parent field: {node.parent.annotation}")
                    print(f"add node.vartype: {node.vartype}")
            else:
                if self.verbose:
                    print(f"skip this Subscript node")
        elif isinstance(node.parent, ast.FunctionDef):
            if node.parent.returns is node:
                node.vartype = ast.unparse(node)
                if self.verbose:
                    print(f"parent field: {node.parent.returns}")
                    print(f"add node.vartype: {node.vartype}")
            else:
                if self.verbose:
                    print(f"skip this Subscript node")
        else:
            if self.verbose:
                print(f"skip this Subscript node")
        return node


class DataTypeChecker(NodeVisitor):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.param_cnt = 0

    def _check_type(self, type_str):
        if type_str in type_class_mapping:
            type_class = type_class_mapping[type_str]
            if self.verbose:
                print(f"type: {type_str}, type class: {type_class}")
            return issubclass(type_class, DataType)
        else:
            if self.verbose:
                print(f"type: {type_str}, type class: unknown class")
            return False

    def visit_Subscript(self, node):
        if self.verbose:
            print(f"\nvisiting {node.__class__.__name__} node...\n")
        if ((isinstance(node.parent, ast.arg) or isinstance(node.parent, ast.AnnAssign)) and node.parent.annotation is node) or \
                (isinstance(node.parent, ast.FunctionDef) and isinstance(node.parent, ast.FunctionDef)):
            if hasattr(node.value, "func"):
                if self._check_type(node.value.func.id):
                    self.param_cnt += 1
                else:
                    raise TypeError('type not supported')
            elif hasattr(node.value, "id"):
                if self._check_type(node.value.id):
                    self.param_cnt += 1
                else:
                    raise TypeError('type not supported')
        else:
            if self.verbose:
                print(f"skip this Subscript node")


if __name__ == "__main__":
    print(f"Type: {bool}, Numpy Type: {bool.numpy_type()}, Numpy Shape: {bool.numpy_shape()}")
    print(f"Type: {int32}, Numpy Type: {int32.numpy_type()}, Numpy Shape: {int32.numpy_shape()}")
    print(f"Type: {uint32}, Numpy Type: {uint32.numpy_type()}, Numpy Shape: {uint32.numpy_shape()}")
    print(f"Type: {int64}, Numpy Type: {int64.numpy_type()}, Numpy Shape: {int64.numpy_shape()}")
    print(f"Type: {uint64}, Numpy Type: {uint64.numpy_type()}, Numpy Shape: {uint64.numpy_shape()}")
    print(f"Type: {float32}, Numpy Type: {float32.numpy_type()}, Numpy Shape: {float32.numpy_shape()}")
    print(f"Type: {float64}, Numpy Type: {float64.numpy_type()}, Numpy Shape: {float64.numpy_shape()}")
    print(f"Type: {complex64}, Numpy Type: {complex64.numpy_type()}, Numpy Shape: {complex64.numpy_shape()}")
    print(f"Type: {complex128}, Numpy Type: {complex128.numpy_type()}, Numpy Shape: {complex128.numpy_shape()}")

    fixed = Fixed(16, 8)[2, 3]
    ufixed = UFixed(16, 8)[2, 3, 4]
    print(f"Type: {fixed}, Numpy Type: {fixed.numpy_type()}, Numpy Shape: {fixed.numpy_shape()}")
    print(f"Type: {ufixed}, Numpy Type: {ufixed.numpy_type()}, Numpy Shape: {ufixed.numpy_shape()}")
