from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SurfaceNumberComponents:
    sign: int
    scale: int
    length: int
    digits: tuple[int, ...]

    def active_digits(self) -> tuple[int, ...]:
        return self.digits[:self.length]

    def digits_str(self) -> str:
        return ''.join(str(d) for d in self.active_digits())


def _to_decimal(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        s = value.strip().replace('_', '')
        try:
            return Decimal(s)
        except InvalidOperation as exc:
            raise ValueError(f"Could not parse numeric string: {value!r}") from exc
    raise TypeError(f"Unsupported numeric value type: {type(value)!r}")


def canonical_decimal_string(value) -> str:
    dec = _to_decimal(value)
    if dec.is_nan() or dec.is_infinite():
        raise ValueError("NaN and infinity are not supported")
    if dec == 0:
        return "0"

    sign_str = "-" if dec < 0 else ""
    body = format(abs(dec), "f")
    if "." in body:
        int_part, frac_part = body.split(".", 1)
        frac_part = frac_part.rstrip("0")
        if frac_part:
            body = f"{int_part}.{frac_part}"
        else:
            body = int_part
    if "." not in body:
        body = body.lstrip("0") or "0"
    if body.startswith("."):
        body = "0" + body
    if body == "":
        body = "0"
    return sign_str + body


def surface_components_from_value(
    value,
    max_digits: int = 32,
    scale_min: int = 0,
    scale_max: int = 32,
) -> SurfaceNumberComponents:
    text = canonical_decimal_string(value)
    sign = 1
    if text.startswith("-"):
        sign = -1
        text = text[1:]

    if text == "0":
        return SurfaceNumberComponents(sign=1, scale=0, length=1, digits=(0,) * max_digits)

    if "." in text:
        int_part, frac_part = text.split(".", 1)
        scale = len(frac_part)
        merged = (int_part + frac_part).lstrip("0")
    else:
        scale = 0
        merged = text.lstrip("0")

    merged = merged or "0"
    if scale < scale_min or scale > scale_max:
        raise ValueError(f"Surface scale {scale} outside supported range [{scale_min}, {scale_max}]")
    if len(merged) > max_digits:
        raise ValueError(f"Surface digit length {len(merged)} exceeds max_digits={max_digits}")

    digits = tuple(int(ch) for ch in merged)
    digits = digits + (0,) * (max_digits - len(digits))
    return SurfaceNumberComponents(sign=sign, scale=scale, length=len(merged), digits=digits)


def render_surface_components(comps: SurfaceNumberComponents) -> str:
    digits = comps.digits_str()
    if digits == "0":
        return "0"

    if comps.scale == 0:
        body = digits
    elif comps.scale >= len(digits):
        zeros = "0" * (comps.scale - len(digits))
        body = f"0.{zeros}{digits}"
    else:
        split = len(digits) - comps.scale
        body = f"{digits[:split]}.{digits[split:]}"

    if "." in body:
        body = body.rstrip("0").rstrip(".")
    if body.startswith("."):
        body = "0" + body
    if body == "":
        body = "0"
    return f"-{body}" if comps.sign < 0 and body != "0" else body


def surface_components_to_row(
    comps: SurfaceNumberComponents,
    max_digits: int,
    scale_min: int = 0,
    scale_max: int = 32,
) -> list[int]:
    if comps.length > max_digits:
        raise ValueError(f"length {comps.length} exceeds max_digits={max_digits}")
    if comps.scale < scale_min or comps.scale > scale_max:
        raise ValueError(f"scale {comps.scale} outside range [{scale_min}, {scale_max}]")
    sign_class = 0 if comps.sign >= 0 else 1
    scale_class = comps.scale - scale_min
    digits = list(comps.digits[:max_digits])
    if len(digits) < max_digits:
        digits.extend([0] * (max_digits - len(digits)))
    return [sign_class, scale_class, comps.length] + digits


def row_to_surface_components(
    row: Sequence[int],
    max_digits: int,
    scale_min: int = 0,
) -> SurfaceNumberComponents:
    sign = 1 if int(row[0]) == 0 else -1
    scale = int(row[1]) + scale_min
    length = int(row[2])
    digits = tuple(int(d) for d in row[3:3 + max_digits])
    if length < 1 or length > max_digits:
        raise ValueError(f"Invalid surface length: {length}")
    return SurfaceNumberComponents(sign=sign, scale=scale, length=length, digits=digits)


def render_surface_row(row: Sequence[int], max_digits: int, scale_min: int = 0) -> str:
    return render_surface_components(row_to_surface_components(row, max_digits=max_digits, scale_min=scale_min))


def surface_rows_from_values(
    values: Iterable,
    max_digits: int = 32,
    scale_min: int = 0,
    scale_max: int = 32,
) -> list[list[int]]:
    rows = []
    for value in values:
        comps = surface_components_from_value(
            value,
            max_digits=max_digits,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        rows.append(surface_components_to_row(
            comps,
            max_digits=max_digits,
            scale_min=scale_min,
            scale_max=scale_max,
        ))
    return rows
