from __future__ import annotations

"""
Analytic number encoder/decoder.

This codec maps a scalar number into a structured analytic representation and
can decode that representation back into a canonical number form.

Design
------
For a nonzero number x, we use canonical scientific form

    x = sign * (d0.d1d2...d_{K-1}) * 10^exp

where:
- sign in {+1, -1}
- d0 in {1,...,9}
- di in {0,...,9} for i >= 1
- K is the fixed number of mantissa digits
- exp is bounded to [exp_min, exp_max]

The representation is the concatenation of:
- sign lane: 2 dims (one-hot)
- exponent lane: analytic Fourier-style encoding over the bounded exponent range
- digit lane: 2 dims per digit using cos/sin on the 10-digit circle
- value lane: optional analytic geometry features for magnitude/context

Notes
-----
- Decoding is exact for the canonicalized representation produced by this codec.
- If the input has more than K significant digits, it is rounded to K digits.
- The value lane is supplemental and is not required for exact decoding.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation, getcontext
from typing import Iterable, List, Sequence

import math
import numpy as np


# Extra precision for Decimal operations during canonicalization.
getcontext().prec = 128


@dataclass(frozen=True)
class NumberComponents:
    sign: int
    exponent: int
    digits: List[int]

    def digits_str(self) -> str:
        return "".join(str(d) for d in self.digits)


class AnalyticNumberCodec:
    def __init__(
        self,
        K: int = 32,
        exp_min: int = -32,
        exp_max: int = 32,
        exponent_periods: Sequence[int] = (65, 13),
        include_value_lane: bool = True,
        log_periods: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
    ) -> None:
        if K <= 0:
            raise ValueError("K must be positive")
        if exp_min > exp_max:
            raise ValueError("exp_min must be <= exp_max")
        if not exponent_periods:
            raise ValueError("Need at least one exponent period")

        self.K = K
        self.exp_min = exp_min
        self.exp_max = exp_max
        self.exponent_periods = tuple(int(p) for p in exponent_periods)
        self.include_value_lane = include_value_lane
        self.log_periods = tuple(float(p) for p in log_periods)

        self.sign_dim = 2
        self.exp_dim = 2 * len(self.exponent_periods)
        self.digit_dim = 2 * self.K
        self.value_dim = 1 + 2 * len(self.log_periods) + 1 if include_value_lane else 0
        self.total_dim = self.sign_dim + self.exp_dim + self.digit_dim + self.value_dim

        self._digit_prototypes = self._make_digit_prototypes()
        self._exp_templates = self._make_exponent_templates()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, value: int | float | str | Decimal) -> np.ndarray:
        comps = self.to_components(value)
        sign_lane = self._encode_sign(comps.sign)
        exp_lane = self._encode_exponent(comps.exponent)
        digit_lane = self._encode_digits(comps.digits)
        if self.include_value_lane:
            value_lane = self._encode_value_lane(comps)
            rep = np.concatenate([sign_lane, exp_lane, digit_lane, value_lane], axis=0)
        else:
            rep = np.concatenate([sign_lane, exp_lane, digit_lane], axis=0)
        return rep.astype(np.float64, copy=False)

    def decode(self, rep: Sequence[float]) -> NumberComponents:
        rep = np.asarray(rep, dtype=np.float64)
        if rep.ndim != 1 or rep.shape[0] != self.total_dim:
            raise ValueError(f"Expected 1D vector of length {self.total_dim}, got shape {rep.shape}")

        idx = 0
        sign_lane = rep[idx : idx + self.sign_dim]
        idx += self.sign_dim
        exp_lane = rep[idx : idx + self.exp_dim]
        idx += self.exp_dim
        digit_lane = rep[idx : idx + self.digit_dim]
        idx += self.digit_dim
        # value lane exists but is not needed for exact decode

        sign = self._decode_sign(sign_lane)
        exponent = self._decode_exponent(exp_lane)
        digits = self._decode_digits(digit_lane)

        # Canonical consistency: if all digits are zero, force zero canonical form.
        if all(d == 0 for d in digits):
            return NumberComponents(sign=1, exponent=0, digits=[0] * self.K)

        # Ensure leading digit is nonzero after decode. If noise caused leading zero,
        # repair by left-normalizing the digit sequence as much as possible.
        if digits[0] == 0:
            digits, exponent = self._renormalize_digits_and_exponent(digits, exponent)

        return NumberComponents(sign=sign, exponent=exponent, digits=digits)

    def roundtrip(self, value: int | float | str | Decimal) -> NumberComponents:
        return self.decode(self.encode(value))

    def to_components(self, value: int | float | str | Decimal) -> NumberComponents:
        d = self._to_decimal(value)
        if d.is_nan() or d.is_infinite():
            raise ValueError("NaN and infinity are not supported")

        if d == 0:
            return NumberComponents(sign=1, exponent=0, digits=[0] * self.K)

        sign = -1 if d < 0 else 1
        d_abs = abs(d)

        exponent = d_abs.adjusted()
        if exponent < self.exp_min or exponent > self.exp_max:
            raise ValueError(
                f"Exponent {exponent} out of supported range [{self.exp_min}, {self.exp_max}]"
            )

        # Scale to [1, 10), then round to K significant digits.
        scaled = d_abs.scaleb(-exponent)
        quant = Decimal(1).scaleb(-(self.K - 1))
        mant = scaled.quantize(quant, rounding=ROUND_HALF_EVEN)

        # Handle rounding overflow, e.g. 9.999... -> 10.000...
        if mant >= Decimal(10):
            mant = mant / Decimal(10)
            exponent += 1
            if exponent < self.exp_min or exponent > self.exp_max:
                raise ValueError(
                    f"Exponent {exponent} out of supported range [{self.exp_min}, {self.exp_max}] after rounding"
                )

        mant_str = format(mant, f".{self.K - 1}f")
        digits_str = mant_str.replace(".", "")
        if len(digits_str) != self.K:
            raise RuntimeError(f"Expected {self.K} mantissa digits, got {digits_str!r}")
        digits = [int(ch) for ch in digits_str]

        # Canonical nonzero numbers must have first digit nonzero.
        if digits[0] == 0:
            raise RuntimeError(f"Canonicalization failed; leading digit is zero for {value!r}")

        return NumberComponents(sign=sign, exponent=exponent, digits=digits)

    def components_to_decimal(self, comps: NumberComponents) -> Decimal:
        if len(comps.digits) != self.K:
            raise ValueError(f"Expected exactly {self.K} digits")
        if all(d == 0 for d in comps.digits):
            return Decimal(0)

        digits_str = comps.digits_str()
        integer_mantissa = Decimal(digits_str)
        shift = comps.exponent - (self.K - 1)
        value = integer_mantissa.scaleb(shift)
        if comps.sign < 0:
            value = -value
        return value

    def components_to_scientific_string(self, comps: NumberComponents, trim_trailing_zeros: bool = False) -> str:
        if all(d == 0 for d in comps.digits):
            return "0"
        sign_str = "-" if comps.sign < 0 else ""
        digits = comps.digits.copy()
        frac = "".join(str(d) for d in digits[1:])
        if trim_trailing_zeros:
            frac = frac.rstrip("0")
        if frac:
            mantissa = f"{digits[0]}.{frac}"
        else:
            mantissa = str(digits[0])
        return f"{sign_str}{mantissa}e{comps.exponent:+d}"

    def components_to_plain_string(self, comps: NumberComponents, trim_trailing_zeros: bool = True) -> str:
        value = self.components_to_decimal(comps)
        s = format(value, "f")
        if trim_trailing_zeros and "." in s:
            s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s

    def decode_to_strings(self, rep: Sequence[float], trim_trailing_zeros: bool = True) -> dict:
        comps = self.decode(rep)
        return {
            "components": comps,
            "scientific": self.components_to_scientific_string(comps, trim_trailing_zeros=trim_trailing_zeros),
            "plain": self.components_to_plain_string(comps, trim_trailing_zeros=trim_trailing_zeros),
            "decimal": str(self.components_to_decimal(comps)),
        }

    def explain_encoding(self, value: int | float | str | Decimal) -> dict:
        comps = self.to_components(value)
        rep = self.encode(value)

        idx = 0
        sign_lane = rep[idx : idx + self.sign_dim]
        idx += self.sign_dim
        exp_lane = rep[idx : idx + self.exp_dim]
        idx += self.exp_dim
        digit_lane = rep[idx : idx + self.digit_dim]
        idx += self.digit_dim
        value_lane = rep[idx:] if self.include_value_lane else np.array([], dtype=np.float64)

        digit_pairs = digit_lane.reshape(self.K, 2)

        return {
            "input": str(value),
            "components": comps,
            "scientific": self.components_to_scientific_string(comps),
            "plain": self.components_to_plain_string(comps),
            "representation_dim": self.total_dim,
            "sign_lane": sign_lane,
            "exp_lane": exp_lane,
            "digit_pairs": digit_pairs,
            "value_lane": value_lane,
            "vector": rep,
        }

    # ------------------------------------------------------------------
    # Internal encoding helpers
    # ------------------------------------------------------------------
    def _encode_sign(self, sign: int) -> np.ndarray:
        if sign >= 0:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array([0.0, 1.0], dtype=np.float64)

    def _encode_exponent(self, exponent: int) -> np.ndarray:
        if exponent < self.exp_min or exponent > self.exp_max:
            raise ValueError(f"Exponent {exponent} out of range")
        feats: list[float] = []
        for p in self.exponent_periods:
            theta = 2.0 * math.pi * exponent / float(p)
            feats.extend([math.cos(theta), math.sin(theta)])
        return np.asarray(feats, dtype=np.float64)

    def _encode_digits(self, digits: Sequence[int]) -> np.ndarray:
        if len(digits) != self.K:
            raise ValueError(f"Expected {self.K} digits")
        feats: list[float] = []
        for d in digits:
            if d < 0 or d > 9:
                raise ValueError(f"Digit out of range: {d}")
            theta = 2.0 * math.pi * d / 10.0
            feats.extend([math.cos(theta), math.sin(theta)])
        return np.asarray(feats, dtype=np.float64)

    def _encode_value_lane(self, comps: NumberComponents) -> np.ndarray:
        value = self.components_to_decimal(comps)
        if value == 0:
            logmag = 0.0
        else:
            logmag = float(abs(value).log10())

        # Normalize exponent to [-1, 1] over the supported exponent range.
        exp_norm = 0.0
        if self.exp_max != self.exp_min:
            mid = 0.5 * (self.exp_max + self.exp_min)
            half_range = 0.5 * (self.exp_max - self.exp_min)
            exp_norm = (comps.exponent - mid) / half_range

        feats: list[float] = [logmag]
        for p in self.log_periods:
            theta = 2.0 * math.pi * logmag / p
            feats.extend([math.cos(theta), math.sin(theta)])
        feats.append(float(exp_norm))
        return np.asarray(feats, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal decoding helpers
    # ------------------------------------------------------------------
    def _decode_sign(self, sign_lane: np.ndarray) -> int:
        return 1 if int(np.argmax(sign_lane)) == 0 else -1

    def _decode_exponent(self, exp_lane: np.ndarray) -> int:
        best_e = None
        best_score = float("inf")
        for e, template in self._exp_templates.items():
            score = float(np.sum((exp_lane - template) ** 2))
            if score < best_score:
                best_score = score
                best_e = e
        if best_e is None:
            raise RuntimeError("Failed to decode exponent")
        return int(best_e)

    def _decode_digits(self, digit_lane: np.ndarray) -> List[int]:
        pairs = digit_lane.reshape(self.K, 2)
        digits: list[int] = []
        for pair in pairs:
            scores = np.sum((self._digit_prototypes - pair[None, :]) ** 2, axis=1)
            digits.append(int(np.argmin(scores)))
        return digits

    def _renormalize_digits_and_exponent(self, digits: List[int], exponent: int) -> tuple[List[int], int]:
        if all(d == 0 for d in digits):
            return [0] * self.K, 0

        shift = 0
        while shift < self.K and digits[shift] == 0:
            shift += 1
        if shift == 0:
            return digits, exponent

        new_digits = digits[shift:] + [0] * shift
        new_exponent = exponent - shift
        if new_exponent < self.exp_min or new_exponent > self.exp_max:
            raise ValueError("Decoded exponent out of range after renormalization")
        return new_digits, new_exponent

    # ------------------------------------------------------------------
    # Template builders
    # ------------------------------------------------------------------
    def _make_digit_prototypes(self) -> np.ndarray:
        prototypes = []
        for d in range(10):
            theta = 2.0 * math.pi * d / 10.0
            prototypes.append([math.cos(theta), math.sin(theta)])
        return np.asarray(prototypes, dtype=np.float64)

    def _make_exponent_templates(self) -> dict[int, np.ndarray]:
        templates: dict[int, np.ndarray] = {}
        for e in range(self.exp_min, self.exp_max + 1):
            templates[e] = self._encode_exponent(e)
        return templates

    # ------------------------------------------------------------------
    # Decimal conversion helper
    # ------------------------------------------------------------------
    def _to_decimal(self, value: int | float | str | Decimal) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError("NaN and infinity are not supported")
            # Convert through string to avoid binary float artifacts.
            return Decimal(str(value))
        if isinstance(value, str):
            s = value.strip().replace("_", "")
            try:
                return Decimal(s)
            except InvalidOperation as exc:
                raise ValueError(f"Could not parse numeric string: {value!r}") from exc
        raise TypeError(f"Unsupported value type: {type(value)!r}")


def _demo() -> None:
    codec = AnalyticNumberCodec(K=32, exp_min=-32, exp_max=32)
    examples = [
        "0",
        "5",
        "-123.45",
        "0.00314",
        "99999999999999999999999999999999",
        "-1.234567890123456789012345678901234e-12",
        "9.99999999999999999999999999999995"
    ]

    print(f"Codec total dim: {codec.total_dim}")
    print()
    for x in examples:
        print(f"Input: {x}")
        comps = codec.to_components(x)
        rep = codec.encode(x)
        decoded = codec.decode(rep)
        print(f"  components.sign     = {comps.sign}")
        print(f"  components.exponent = {comps.exponent}")
        print(f"  components.digits   = {comps.digits_str()}")
        print(f"  components          = {comps}")
        print(f"  encoded             = {rep}")
        print(f"  decoded             = {decoded}")
        print(f"  scientific          = {codec.components_to_scientific_string(decoded)}")
        print(f"  plain               = {codec.components_to_plain_string(decoded)}")
        print(f"  rep[:16]            = {np.array2string(rep[:16], precision=4, suppress_small=True)}")
        print()


if __name__ == "__main__":
    _demo()
