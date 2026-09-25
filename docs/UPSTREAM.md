# Upstream fixes that would remove polars-pylance workarounds

Status as of 2026-09-25. polars-pylance requires `pylance>=12`; pylance 12.0.0 and
Lance `main` build on DataFusion 54 (the move to 55 is being prepared in
[lance#8997](https://github.com/lance-format/lance/pull/8997)). Code links point at
`main` @ `2145054`.

| Workaround | Code in polars-pylance | Upstream issue | Upstream PR | Fixed in |
| --- | --- | --- | --- | --- |
| **Zero literals.** Lance compared `-0.0` and `0.0` unequal against a zero literal, so every float `=`, ordering, `IN` and `BETWEEN` against zero spelled out both zeros, and the integer `%` / `//` sign correction used a long `(r < 0 AND b > 0) OR …` form. | Removed in [#41](https://github.com/jonasdedden/polars-pylance/pull/41). | none | [lance#6236](https://github.com/lance-format/lance/pull/6236) | **pylance 12.0.0**, already required |
| **NaN sign.** Lance orders a NaN with its sign bit set (what `0 / 0` gives on x86) below `-inf`; Polars puts every NaN above every number. Against a literal, `x < -inf` is added to or excluded from ordering comparisons; between two float values, `nanvl(x, NaN)` gives every NaN the positive sign. | [`_float_comparison`, L1183-1191][pred-nan] | [lance#9315](https://github.com/lance-format/lance/issues/9315) | [lance#9324](https://github.com/lance-format/lance/pull/9324) | Not yet (PR open) |
| **Signed zeros between two float values.** Column-to-column comparisons use Arrow's total order, so `-0.0 = 0.0` is false. The zero pair is decided explicitly: `… OR (x = 0 AND y = 0)` / `… AND NOT (…)`. | [`_float_comparison`, L1194][pred-zeros] | [lance#9316](https://github.com/lance-format/lance/issues/9316) | [lance#9323](https://github.com/lance-format/lance/pull/9323); possibly also [datafusion#22835](https://github.com/apache/datafusion/pull/22835) (unverified for Lance) | Not yet. datafusion#22835 is in DataFusion 55.0.0, which no pylance uses yet |
| **`array_has` with a zero.** `array_has(l, 0.0)` does not find `-0.0`. `list.contains(0.0)` is sent as `array_has(l, -0.0) OR array_has(l, 0.0)`. | [`_contains`, L579-584][pred-contains] | [lance#9316](https://github.com/lance-format/lance/issues/9316) | [lance#9323](https://github.com/lance-format/lance/pull/9323) | Not yet (PR open) |
| **Float literal against an integer column.** Lance refuses `i > 1.5` ("could not convert to literal of type Int64"). The integer side is cast to `double`, which costs its scalar index. | [`_compare`, L402-408][pred-compare] | [lance#9317](https://github.com/lance-format/lance/issues/9317) | [lance#9333](https://github.com/lance-format/lance/pull/9333) | Not yet (PR open) |
| **Float64 literal against a Float32 column.** Lance narrows a Float64 literal to a bare Float32 column, where Polars widens the column. The column is cast to `double` instead (comparisons and float `is_in` lists), and an integer `is_in` list over a float column is cast to Float64 first. The cast costs the column's scalar index; narrowing only exactly representable literals upstream would let Lance keep it. | [`_compare`, L405-408][pred-compare]; [`_is_in`, L500 and L514][pred-isin] | [lance#9318](https://github.com/lance-format/lance/issues/9318) | [lance#9512](https://github.com/lance-format/lance/pull/9512) | Not yet (PR open) |
| **`xor`.** `flag != (id > 0)` fails to plan, because Lance converts every literal on the right of a boolean column to Boolean. `a ^ b` is expanded to `(a AND NOT b) OR (NOT a AND b)`, and declines unless both sides are exact. | [`_binary_predicate`, L375-387][pred-xor] | [lance#9319](https://github.com/lance-format/lance/issues/9319) | [lance#9322](https://github.com/lance-format/lance/pull/9322) | Not yet (PR open) |
| **`eq_missing` / `ne_missing`.** Lance rejects `IS [NOT] DISTINCT FROM`, so null-safe equality is only lowered against a non-null literal, as `(x = v) IS TRUE` / `IS NOT TRUE`; between two columns it declines. | [`_NULL_SAFE`, L53][pred-nullsafe-const]; [`_binary_predicate`, L367-373][pred-nullsafe] | none | [lance#9331](https://github.com/lance-format/lance/pull/9331) | Not yet (PR open) |
| **`x ** 0`.** DataFusion's simplifier folds `power(x, 0)` to `1`, turning null rows into `1`. A zero or negative exponent declines. | [`_power`, L786][pred-power] | [datafusion#24246](https://github.com/apache/datafusion/issues/24246) | [datafusion#24247](https://github.com/apache/datafusion/pull/24247) | **DataFusion 55.0.0**; no pylance release yet (Lance is on DataFusion 54) |
| **Float `//`.** DataFusion's one-argument `trunc` over an array maps `-0.0` to `0.0` (`if x == 0_f64 { 0_f64 } else { x.trunc() }` in `math/trunc.rs`), while its scalar path and `trunc(x, 0)` keep the sign. Float floor division declines. (`trunc(x, 0)` keeps the sign on pylance 12.0.0 and may work around it; not swept.) | [`_remainder_or_floor`, L683][pred-floordiv] (float operands fail the integer check) | [datafusion#25702](https://github.com/apache/datafusion/issues/25702) | [datafusion#25732](https://github.com/apache/datafusion/pull/25732) | Not yet (PR open); then needs a DataFusion release and a Lance upgrade to it |
| **No `FLOOR` / `CEIL` in Lance filters.** Lance's planner has no branch for `sqlparser`'s `Floor` / `Ceil` forms ("is not supported SQL in lance"). Float `%` spells Polars' `a - b * floor(a / b)` as `trunc(q) - CAST(trunc(q) - q > 0.0 AS …)`; `FLOOR` would also give float `//` a direct spelling. | [`_float_modulus`, L712][pred-floatmod] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | none yet | Not yet (no PR) |
| **No `CASE` in Lance filters.** Same planner gap. `when/then/otherwise` declines. | [`_predicate`, L339][pred-case] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | none yet | Not yet (no PR) |
| **No `SUBSTRING` in Lance filters.** Same planner gap. `str.slice` declines (it has no branch in the lowering). | [`docs/PUSHDOWN.md`, L119][docs-slice] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | none yet | Not yet (no PR) |

[pred-nan]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L1183-L1191
[pred-zeros]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L1194
[pred-contains]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L579-L584
[pred-compare]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L402-L408
[pred-isin]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L498-L514
[pred-xor]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L375-L387
[pred-nullsafe-const]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L53
[pred-nullsafe]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L367-L373
[pred-power]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L773-L791
[pred-floordiv]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L664-L692
[pred-floatmod]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L694-L714
[pred-case]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L339-L342
[docs-slice]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/docs/PUSHDOWN.md?plain=1#L119

## Not upstream bugs

These declines follow from Lance/DataFusion behaving as SQL or IEEE 754 specify, or
from deliberate choices, so there is nothing to file:

- `round` breaks ties away from zero (SQL), Polars to even.
- `(-inf) ** 0.5` is `inf` (IEEE 754 `pow`), NaN in Polars.
- `0 ** -1` and `i64::MIN // -1` fail the scan (as in Postgres); Polars gives `inf` / wraps.
- A float cast to string prints `1e16`, Polars `1e+16`.
- `utf8 + utf8` is rejected; text concatenation is `||`.
- UInt64 with a signed integer is coerced to `Decimal128(20, 0)`
  ([datafusion#14223](https://github.com/apache/datafusion/pull/14223)); the UInt64
  `//` decline comes from the lowering mixing in a `bigint` correction term.
- `if()` and `mod()` are not DataFusion functions; `%` works.
- Unclear, not investigated: `ln` of a Float32 and `log10` are an ulp off.
