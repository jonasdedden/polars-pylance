# Upstream fixes that would remove polars-pylance workarounds

Status as of 2026-10-28. polars-pylance requires `pylance>=14`. The latest stable pylance
is [14.0.1](https://github.com/lance-format/lance/releases/tag/v14.0.1); 15.0.0 is in
beta (`15.0.0b4`, from `https://pypi.fury.io/lance-format/`). Every Lance release up to
15.0.0-beta.4 builds on DataFusion 54.1.0. Lance `main` moved to DataFusion 55.2.0 with
[lance#10397](https://github.com/lance-format/lance/pull/10397) on 2026-10-28, and the
15.0.0 betas are cut from `main`. Code links point at `main` @ `f058bc6`.

Every claim below was checked against pylance 14.0.0, 14.0.1, 15.0.0b3 and 15.0.0b4 by
running the SQL against data holding NaN of both signs, signed zeros, infinities,
integers past 2^53 and multi-byte strings, split so that some zones hold only NaN. Rows
were compared with Polars plain and under `NOT`, without an index and with BTREE,
ZONEMAP and BITMAP indices, and `explain_plan()` was read for scalar index use.

## Removable with pylance 15

Lance fixed these in the 15.0.0 betas; none is in a stable release yet. Requiring
`pylance>=15` turns each comparison into the plain one. Prototyped on `f058bc6`, that
removes 93 lines of `_predicate.py` and adds 6. On 15.0.0b4 only the tests that pin the
old SQL text fail; on 14.0.1, 121 row-by-row tests fail, so the floor has to move with
the code.

| Workaround | Code | Upstream | Fixed in |
| --- | --- | --- | --- |
| **NaN sign against a literal.** Lance orders a NaN with its sign bit set (what `0 / 0` gives on x86) below `-inf`; Polars puts every NaN above every number. `>` / `>=` add `x < -inf`, which holds exactly for a negative NaN. Pylance 15 adds that term itself (`x > 1.0` plans as `OR([x > 1], [x < -inf])` on the index), also for computed and Float32 values. | [`_float_comparison`, L1298-1309][pred-literal] | [lance#9315](https://github.com/lance-format/lance/issues/9315), fixed by [lance#9628](https://github.com/lance-format/lance/pull/9628) | 15.0.0-beta.2 |
| **Zone maps skip NaN-only zones.** A ZONEMAP index treats a zone holding only positive NaNs as empty, so `>` / `>=` add `x = NaN`, and `<` / `<=` use `NOT isnan(x)` rather than `x >= -inf`, which under `NOT` would become such a range again. Both keep a BTREE index. Pylance 15 reads zone maps written by 14 correctly too. | [`_float_comparison`, L1298-1309][pred-literal] | [lance#10102](https://github.com/lance-format/lance/pull/10102) | 15.0.0-beta.1 |
| **NaN sign and signed zeros between two float values.** Arrow's total order puts `-0.0` below `0.0` and a negative NaN below everything. `nanvl(x, NaN)` gives every NaN the positive sign, and the pair of zeros is decided explicitly. Pylance 15 compares through its own `_lance_compare_floats`, which ignores both signs; all six operators match Polars on 15.0.0b4, computed values and Float32 against Float64 included. ([datafusion#22835](https://github.com/apache/datafusion/pull/22835) in DataFusion 55 normalizes zeros only, not NaN, and does not matter here.) | [`_float_comparison`, L1311-1320][pred-columns] | [lance#9315](https://github.com/lance-format/lance/issues/9315), [lance#9316](https://github.com/lance-format/lance/issues/9316), fixed by [lance#9628](https://github.com/lance-format/lance/pull/9628) | 15.0.0-beta.2 |
| **`array_has` with a zero.** `array_has(l, 0.0)` does not find `-0.0`, so `list.contains(0.0)` is sent as `array_has(l, -0.0) OR array_has(l, 0.0)`. Pylance 15 rewrites a zero needle to `array_has_any` over both encodings, for Float64 and Float32 lists. | [`_contains`, L633-638][pred-contains] | [lance#9316](https://github.com/lance-format/lance/issues/9316), fixed by [lance#9628](https://github.com/lance-format/lance/pull/9628) | 15.0.0-beta.2 |
| **`nan_free` tracking.** Each `_Value` records whether it can be NaN, so that the guards above are left off where they cannot matter. Without the guards, only `min_horizontal` / `max_horizontal` read it, and wrapping every float there in `nanvl(x, NULL)` costs nothing. | [`nan_free`, L162-164][pred-nanfree]; [`may_be_nan`][pred-maybenan], [`cannot_be_nan`][pred-cannotbenan] and their callers | follows from the rows above | 15.0.0-beta.2 |

Pylance 15 also lets more be pushed, which would add code rather than remove it:

- `eq_missing` / `ne_missing` between two floats as `IS [NOT] DISTINCT FROM`
  ([lance#9331](https://github.com/lance-format/lance/pull/9331) in 14, correct for
  floats only with lance#9628). Declined today in [`_null_safe`][pred-nullsafe].
- NaN in an `is_in` list: `x IN (CAST('NaN' AS double), …)` matches Polars on 15 and
  not on 14. It declines in [`_scalar`][pred-scalar]. An infinity in a list needs no
  upstream fix and is lowered by
  [#64](https://github.com/jonasdedden/polars-pylance/pull/64). A scalar NaN literal,
  and so `x = NaN` or `list.contains(NaN)`, cannot be pushed whatever the pylance version
  until Polars changes; see *Non-finite float literals* below.

## Waiting on further upstream changes

These stay even with pylance 15.

| Workaround | Code | What upstream would have to change |
| --- | --- | --- |
| **Float literal against an integer column.** Lance up to 14 refuses `i > 1.5`, so the integer side is cast to `double`, which costs its scalar index. Pylance 15 accepts the bare form ([lance#9333](https://github.com/lance-format/lance/pull/9333)) but compares the integer exactly, where Polars converts it to Float64 first: on 15.0.0b4, `i > 9007199254740992.0`, `i = 2^53`, `i < -2^53` and `i IN (2^53)` return different rows for `i = 2^53 + 1`. So with pylance 15 the cast could only go for a literal below 2^53 in magnitude or a column of at most 32 bits, and Polars' optimizer has already put `TRY_CAST(i AS double)` around the column, which the lowering would have to strip. | [`as_double`, L209-228][pred-asdouble] via [`_compare`, L452-458][pred-compare]; [`_is_in`, L565-568][pred-isin] | Turn `CAST(int AS double) op literal`, and `TRY_CAST`, into an index range on the integer when the literal is below 2^53, the follow-up #9333 left open. polars-pylance would keep its spelling, which matches Polars, and get the index back. |
| **Float64 literal against a Float32 column.** Lance narrows a bare float literal to a Float32 column, where Polars widens the column: `f > 0.1` returns wrong rows, still on 15.0.0b4. So the literal is sent typed, `f > CAST(0.1 AS double)`, which [lance#9512](https://github.com/lance-format/lance/pull/9512) (14.0.0) narrows only where exact and otherwise compares in double. The `cast(Float64)` Polars' optimizer puts around the column is stripped first, so the index still applies. | [`_compare`, L452-458][pred-compare]; [`_is_in`, L565-568][pred-isin]; [`widened`, L165-167][pred-widened] and [L719][pred-widenedset] | Two changes, one per half: narrowing a bare literal only where exact, as #9512 does for a typed one, would drop the typed spelling; turning `CAST(f AS double) op literal` (and `TRY_CAST`) into an index range on `f` when the literal is exact in Float32 would drop the stripping. |
| **Non-finite float literals.** `Expr.meta.serialize(format="json")`, which the lowering reads, writes `inf`, `-inf` and NaN as `null`, for a bare literal (`{"Dyn": {"Float": null}}`) and a typed one (`{"Scalar": {"Float64": null}}`) alike, so the value is gone before the lowering sees it and the literal declines. Only `is_in` lists keep it, being Arrow IPC. With the value, `x > -inf`, `x < inf`, `x = inf` and the rest would be correct on pylance 14 already, through the NaN guards above (checked on 14.0.1 with every index type), and `x = NaN` and `list.contains(NaN)` from pylance 15. | [`_literal_series`, L1144-1148][pred-literalseries] | A Polars change: write non-finite floats into the JSON IR distinguishably, for example as a string or a bit pattern, or offer a way to walk an expression that does not go through JSON. |
| **`xor` as an expansion.** `a ^ b` is sent as `(a AND NOT b) OR (NOT a AND b)`. `a != b` plans since pylance 14 ([lance#9322](https://github.com/lance-format/lance/pull/9322)), but never reaches a scalar index, while the expansion does: `(i > 2) ^ (i < 6)` becomes `OR([i >= 6], [i <= 2])`. | [`_binary_predicate`, L405-415][pred-xor] | Lance's index planner expanding `bool != bool` into index searches itself. |
| **`eq_missing` / `ne_missing` against a literal** are sent as `(x = v AND x IS NOT NULL)` / `(NOT (x = v) OR x IS NULL)`, which reach the index. `x IS NOT DISTINCT FROM v` works since pylance 14 but uses no index, on 15.0.0b4 either. | [`_NULL_SAFE`, L54-57][pred-nullsafe-const] | Lance's index planner reading `x IS NOT DISTINCT FROM v` as `[x = v]`. The spelling would be no shorter, so this is cosmetic. |
| **No zero exponent.** DataFusion 54's simplifier folds `power(x, 0)` to `1`, turning null rows into `1`, so `x ** 0` declines. Polars computes `powf(x, 0.0)` there, as Lance would. | [`_power`, L887][pred-power] | [datafusion#24247](https://github.com/apache/datafusion/pull/24247) folds only a non-nullable base, from DataFusion 55.0.0. It is in Lance `main` through [lance#10397](https://github.com/lance-format/lance/pull/10397), in no release yet; then allow exponent 0. |
| **`CASE`, `FLOOR`, `CEIL` and `SUBSTRING` are syntax Lance's planner rejects.** `floor`, `ceil` and `substr` are sent backtick-quoted, which makes them plain DataFusion calls. `when/then/otherwise` is sent as `(c IS TRUE AND a) OR (c IS NOT TRUE AND b)`, and as a value as `nvl2(NULLIF(CAST(c IS TRUE AS bigint), 0), a, b)`: DataFusion 54 coerces `nvl2`'s first argument together with the branches, so a Boolean there fails to plan. | [`_ternary`, L364-375][pred-ternary]; [`_ternary_value`, L671-697][pred-ternaryvalue]; [`floor` / `ceil`, L810-821][pred-floorceil]; [`_float_floor`, L796][pred-floatfloor]; [`_slice`, L958-974][pred-slice] | [lance#9584](https://github.com/lance-format/lance/pull/9584) (open, conflicting) for [lance#9560](https://github.com/lance-format/lance/issues/9560) would let them be written directly; a DataFusion `nvl2` that coerces only its branches would drop the `NULLIF(CAST(… AS bigint), 0)`. Cosmetic: the rows are the same either way. |

## Declines that are not upstream bugs

These follow from Lance/DataFusion behaving as SQL or IEEE 754 specify, from Polars'
own choices, or from Lance's typing, so there is nothing to file:

- `round` breaks ties away from zero (SQL), Polars to even.
- `power` is `powf` in Float64. Polars raises to an integer exponent in wrapping integer
  arithmetic and a Float32 in Float32, takes `sqrt` for 0.5 (so `(-inf) ** 0.5` is NaN
  there and `inf` in Lance), and multiplies out whole exponents from 2 to 9, an ulp off
  `powf` on up to half the rows. Those decline; other Float64 exponents of an integer or
  Float64 base match bit for bit.
- `0 ** -1` fails the scan ("zero raised to a negative power is undefined", as in
  Postgres), even for a float base, where IEEE `pow` and Polars give `inf`. Pushing a
  negative exponent would need Arrow's `power` kernel to return `±inf` instead.
- `i64::MIN // -1` fails the scan; Polars wraps.
- `%` and `//` by a fractional float literal: Polars' column-by-literal kernels disagree
  with its column-by-column ones, `0.3 // 0.1` is `3.0` where `floor(0.3 / 0.1)` is
  `2.0`, and on a one-row frame Polars itself gives `2.0`.
- A UInt64 with a signed integer is coerced to `Decimal128(20, 0)`
  ([datafusion#14223](https://github.com/apache/datafusion/pull/14223)), so their
  quotient is a `Decimal128(24, 4)` with a fraction: `u / NULLIF(3, 0)` gives `3.3333`
  for 10, and even `(u / 3) = 3` inside a filter divides as a decimal. UInt64 `//` and
  `eq_missing` between a UInt64 and a signed integer decline.
- `when/then/otherwise` as a value whose branches differ in type: Lance would pick the
  result type by its own rules.
- A float cast to string prints `1e16`, Polars `1e+16`.
- `utf8 + utf8` is rejected; text concatenation is `||`.
- `if()` and `mod()` are not DataFusion functions; `%` works.
- A bare `NULL` filter is rejected ("does not return a boolean"), hence
  `CAST(NULL AS boolean)`; `IN ()` is a parse error in `sqlparser`.
- A double-quoted identifier is not rejected: `"i"` parses as the string `'i'`, so
  `"i" = 3` fails to cast and `"s" = 's'` would be a constant comparison. Identifiers
  are backtick-quoted ([`_quote`][pred-quote]).
- Unclear, not investigated: `ln` of a Float32 and `log10` are an ulp off.

[pred-literal]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L1298-L1309
[pred-columns]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L1311-L1320
[pred-contains]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L633-L638
[pred-nanfree]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L162-L164
[pred-maybenan]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L192-L194
[pred-cannotbenan]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L201-L204
[pred-nullsafe]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L429-L436
[pred-scalar]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L1206-L1208
[pred-literalseries]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L1144-L1148
[pred-asdouble]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L209-L228
[pred-compare]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L452-L458
[pred-isin]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L565-L568
[pred-widened]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L165-L167
[pred-widenedset]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L719
[pred-xor]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L405-L415
[pred-nullsafe-const]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L54-L57
[pred-power]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L887
[pred-ternary]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L364-L375
[pred-ternaryvalue]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L671-L697
[pred-floorceil]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L810-L821
[pred-floatfloor]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L796
[pred-slice]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L958-L974
[pred-quote]: https://github.com/jonasdedden/polars-pylance/blob/f058bc60c0e29244ddc1970be9d1dd0d490c1df7/src/polars_pylance/_predicate.py#L1130-L1134
