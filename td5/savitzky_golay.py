"""Savitzky Golay demo for sysnum TD"""

import functools
import math
import typing

import sympy as sp  # type:ignore

import lib_carotte

try:
    from lib_carotte import *
except ImportError:
    from .lib_carotte import *  # type: ignore # pylint: disable=relative-beyond-top-level


class Utils:
    """Misc utils"""

    @staticmethod
    def zero(size: int) -> Variable:
        "n-bit zero constant"
        return Constant("0" * size)

    @staticmethod
    def ones(size: int) -> Variable:
        "n-bit ones constant"
        return Constant("1" * size)

    @staticmethod
    def truncate(a: Variable, size: int, signed: bool = False) -> Variable:
        """Resize `a` to `size` bits. If `a` needs be extended, `signed` toggles zero/sign-extension."""
        assert size > 0
        d = size - len(a)
        if d > 0:
            expand = Mux(a[len(a) - 1], Utils.zero(d), Utils.ones(d)) if signed else Utils.zero(d)
            return a + expand
        if d < 0:
            return a[:size]
        return a

    @staticmethod
    def constant(n: int, v: int) -> Variable:
        """n-bit constant from an integer constant"""
        return Utils.truncate(Constant(bin(v + 2 ** max(v.bit_length(), n))[2:]), n, signed=True)

    @staticmethod
    def nnot(a: Variable) -> Variable:
        """n-bit Not, for those without `allow_ribbon_logic_operations(True)`"""
        if lib_carotte._ALLOW_RIBBON_LOGIC_OPERATIONS:  # pylint: disable=protected-access
            return Not(a)
        na = functools.reduce(lambda x, y: y if x is None else x + y, [~x for x in a], None)
        assert na is not None
        return na

    @staticmethod
    def nreg(a: Variable) -> Variable:
        """n-bit register, for those without `allow_ribbon_logic_operations(True)`"""
        if lib_carotte._ALLOW_RIBBON_LOGIC_OPERATIONS:  # pylint: disable=protected-access
            return Reg(a)
        nreg = functools.reduce(lambda x, y: y if x is None else x + y, [Reg(x) for x in a], None)
        assert nreg is not None
        return nreg

    @staticmethod
    def full_adder(a: Variable, b: Variable, c: typing.Optional[Variable] = None) -> typing.Tuple[Variable, Variable]:
        """1-bit full adder implementation"""
        tmp = a ^ b
        return (tmp, a & b) if c is None else (tmp ^ c, (tmp & c) | (a & b))

    _STATS_NADDER = 0

    @staticmethod
    def nadder(
        n: int, a: Variable, b: Variable, c_in: typing.Optional[Variable] = None
    ) -> typing.Tuple[Variable, Variable]:
        """n-bit full-adder. Returns a+b"""
        Utils._STATS_NADDER += n
        assert len(a) == len(b) == n
        return functools.reduce(
            lambda x, y: (  # pylint: disable=unnecessary-direct-lambda-call
                lambda r=Utils.full_adder(y[0], y[1], x[1]): (  # type: ignore
                    r[0] if x[0] is None else x[0] + r[0],  # type: ignore
                    r[1],  # type: ignore
                )
            )(),
            zip(a, b),
            (None, c_in),
        )

    @staticmethod
    def shift_left(a: Variable, s: int) -> Variable:
        """Shift `a` left by `s` units, adding zeroes as needed"""
        assert s >= 0
        return Utils.zero(s) + a if s > 0 else a

    @staticmethod
    def shift_right(a: Variable, s: int) -> Variable:
        """Logically shift `a` right by `s` units. If `s>=len(a)`, a 1-bit zero is returned"""
        assert s >= 0
        return Constant("0") if s >= len(a) else a[s:] if s > 0 else a

    @staticmethod
    def mult(n: int, a: Variable, b: int) -> Variable:
        """Basic n-bit shift-add multiplier. Returns a×b"""
        ma = Utils.truncate(a, n, signed=True)
        if b == 1:
            return ma
        ms = Utils.constant(n, -(b < 0))
        for i in range(n):
            if (abs(b) >> i) & 1:
                ms, _ = Utils.nadder(n, ms, Utils.truncate(Utils.shift_left(ma, i), n))
        return ms if b >= 0 else Utils.nnot(ms)

    @staticmethod
    def is_power_of_two(n: int) -> bool:
        """Classic bit trick for power of 2"""
        return (n != 0) and (n & (n - 1) == 0)

    @staticmethod
    def compute_div_constant(n: int, d: int) -> typing.Tuple[int, int, int]:
        """Get the parameters for division by constant 'd' of integers of size 'n'"""
        assert d > 0
        l = d.bit_length() - 1

        if Utils.is_power_of_two(d):
            out_mul, out_add = (1, 0)
        else:
            m_down = (1 << (n + l)) // d
            m_up = m_down + 1
            temp = m_up * d

            if temp <= (1 << l):
                out_mul, out_add = (m_up, 0)
            else:
                out_mul = out_add = m_down

        out_shift = l
        return (out_mul, out_add, out_shift)

    @staticmethod
    def div(n: int, a: Variable, d: int) -> Variable:
        """Simple n-bit division by a constant. Returns floor(a/d)"""
        assert d > 0
        a = Utils.truncate(a, n, signed=True)
        if d == 1:
            return a
        mul, add, shift = Utils.compute_div_constant(n, d)
        if mul == 1 and add == 0:
            s = a
        else:
            s, _ = Utils.nadder(2 * n, Utils.mult(2 * n, a, mul), Utils.constant(2 * n, add))
        return Utils.truncate(Utils.shift_right(s, n + shift), n, signed=True)


class SavitzkyGolay:
    """Savitzky Golay implementation"""

    IN_BITS = 3
    COMPUTE_BITS = 16

    @staticmethod
    def compute_coeffs(window_size: int, poly_order: int, deriv: int) -> typing.Tuple[list[int], int]:
        """Generate coefficients for the Savitzky Golay algorithm."""
        assert window_size > 0 and window_size % 2 == 1 and window_size + deriv > poly_order + 1
        hw = (window_size - 1) // 2
        b = sp.matrices.Matrix([[sp.Rational(k**i) for i in range(poly_order + 1)] for k in range(-hw, hw + 1)])
        c = ((b.T * b).inv() * b.T).row(deriv) * math.factorial(deriv)
        normalization = math.lcm(*[sp.fraction(x)[1] for x in c])
        coeffs = [int(x * normalization) for x in c]
        return coeffs, normalization

    def smoothing(self, values: list[Variable], coeffs: list[int], normalization: int) -> Variable:
        """Generates the filter, r = sum_i(coeffs[i]*values[i])/normalization"""
        assert len(values) == len(coeffs)

        s = Utils.zero(self.COMPUTE_BITS)  # Accumulator

        for val, coeff in zip(values, coeffs):
            if coeff != 0:
                e = Utils.mult(self.COMPUTE_BITS, val, coeff)
                s, _ = Utils.nadder(self.COMPUTE_BITS, s, e)

        # Apply normalization
        r = Utils.div(self.COMPUTE_BITS, s, normalization)
        return r

    def verif_smoothing(self, values: list[Variable], coeffs: list[int], normalization: int, r: Variable) -> None:
        """To be filled with the formal model; see TD-Verif question 🍅"""
        pass

    def __init__(self) -> None:
        POLY_ORDER = 3
        WINDOW_SIZE = 5
        DERIV_ORDER = 0

        coeffs, normalization = self.compute_coeffs(WINDOW_SIZE, POLY_ORDER, DERIV_ORDER)
        Utils._STATS_NADDER = 0

        print("Generating a Savitzky Golay circuit with:")
        print(f"  {POLY_ORDER=:2} {WINDOW_SIZE=:2} {DERIV_ORDER=:2}")
        print(f"  Generated: {coeffs=} {normalization=}")

        in_v = Input(self.IN_BITS)

        r: list[Variable] = [in_v]
        for i in range(len(coeffs) - 1):
            x = Utils.nreg(r[i])
            x.rename("in_" + "v" * (i + 2))
            r.append(x)

        out = self.smoothing(r, coeffs, normalization)
        out.set_as_output("out")
        self.verif_smoothing(r, coeffs, normalization, out)

        print(f"Generated circuit has GATE-SCORE: {Utils._STATS_NADDER}")


def main() -> None:
    """Savitzky Golay demo"""  # https://youtu.be/pZazxFu087M https://youtu.be/ipxpC-IcBvM
    # allow_ribbon_logic_operations(True) # Please uncomment if your simulator supports this
    SavitzkyGolay()


if __name__ == "__main__":
    raise RuntimeError("You need to run this file with carotte.py.")
