import sar_dsl
from frontend.type_opt import complex64


def app(
    sig1: complex64[2, 3],
    sig2: complex64[2, 3]
) -> complex64[3, 3]:

    sig3: complex64[3, 3]
    sig3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    sig3[:2, :] = sig1 * sig2

    return sig3


if __name__ == "__main__":
    sar_dsl.build(app)
