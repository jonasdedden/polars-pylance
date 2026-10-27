# Upstream fixes that would remove polars-pylance workarounds

Status as of 2026-10-27. polars-pylance requires `pylance>=12`. The latest stable
pylance is [14.0.0](https://github.com/lance-format/lance/releases/tag/v14.0.0)
(2026-10-26); 15.0.0 is in beta (`15.0.0b3`, from `https://pypi.fury.io/lance-format/`).
Every Lance release up to 15.0.0-beta.3 and Lance `main` still build on DataFusion
54.1.0. [lance#8997](https://github.com/lance-format/lance/pull/8997) only prepared the
move to 55; the upgrade itself is
[lance#9979](https://github.com/lance-format/lance/pull/9979) (open). Code links point
at `main` @ `2145054`.

Every row below was checked by running both the SQL the lowering emits and the naive
SQL it avoids against pylance 12.0.0, 14.0.0 and 15.0.0b3, comparing the rows with
Polars and reading `explain_plan()` for scalar index use. The test suite passes
unchanged on all three versions.

## What each pylance floor unlocks

| Floor | Change in the lowering |
| --- | --- |
| none (works on pylance 12) | `floor`, `ceil` and `substr` as backtick-quoted calls; float `//` via `floor`; an indexable `eq_missing` / `ne_missing` against a literal; `eq_missing(None)` as `IS NULL`. See [Possible without an upstream fix](#possible-without-an-upstream-fix). |
| `pylance>=14` | A Float64 literal against a Float32 column keeps the scalar index (cast the literal, not the column). `eq_missing` / `ne_missing` between two non-float values. |
| `pylance>=15` | Drop the NaN-sign and signed-zero corrections and the zero `array_has` disjunction. Compare integer columns with float literals directly, keeping the index. `eq_missing` / `ne_missing` between two float values. NaN literals can be pushed. |
| Lance on DataFusion 55 | `x ** 0`. |
| Lance on DataFusion 56 | One-argument `trunc` keeps `-0.0`. Not needed if float `//` uses `floor`. |
| [lance#9584](https://github.com/lance-format/lance/pull/9584) merged | `when/then/otherwise` as `CASE`. |

## Workarounds

| Workaround | Code in polars-pylance | Upstream issue | Upstream PR | Fixed in | Then |
| --- | --- | --- | --- | --- | --- |
| **Zero literals.** Lance compared `-0.0` and `0.0` unequal against a zero literal, so every float `=`, ordering, `IN` and `BETWEEN` against zero spelled out both zeros, and the integer `%` / `//` sign correction used a long `(r < 0 AND b > 0) OR …` form. | Removed in [#41](https://github.com/jonasdedden/polars-pylance/pull/41). | none | [lance#6236](https://github.com/lance-format/lance/pull/6236) | **pylance 12.0.0**, already required | Done. |
| **NaN sign.** Lance orders a NaN with its sign bit set (what `0 / 0` gives on x86) below `-inf`; Polars puts every NaN above every number. Against a literal, `x < -inf` is added to or excluded from ordering comparisons; between two float values, `nanvl(x, NaN)` gives every NaN the positive sign. | [`_float_comparison`, L1178-1191][pred-nan] | [lance#9315](https://github.com/lance-format/lance/issues/9315) (closed) | [lance#9628](https://github.com/lance-format/lance/pull/9628), which superseded [lance#9324](https://github.com/lance-format/lance/pull/9324) (closed unmerged) | **pylance 15.0.0-beta.2**; no stable release yet | With `pylance>=15`, send the plain comparison. Lance now adds `x < -inf` itself (`x > 1.0` plans as `OR([x > 1], [x < -inf])` on the index), and compares two columns ignoring the NaN sign. Until then the guard stays and is harmless on 15: Lance rewrites the inner `x < -inf` to exclude NaN and adds its own. |
| **Signed zeros between two float values.** Column-to-column comparisons use Arrow's total order, so `-0.0 = 0.0` is false. The zero pair is decided explicitly: `… OR (x = 0 AND y = 0)` / `… AND NOT (…)`. | [`_float_comparison`, L1194-1197][pred-zeros] | [lance#9316](https://github.com/lance-format/lance/issues/9316) (closed) | [lance#9628](https://github.com/lance-format/lance/pull/9628), which superseded [lance#9323](https://github.com/lance-format/lance/pull/9323) (closed unmerged) | **pylance 15.0.0-beta.2**; no stable release yet | With `pylance>=15`, drop the zero pair together with `nanvl`: `x = y`, `x != y`, `x < y` and `x >= y` all match Polars on 15.0.0b3. [datafusion#22835](https://github.com/apache/datafusion/pull/22835) (DataFusion 55) does not matter here. Lance compares through its own `_lance_compare_floats` UDF, and #22835 normalizes zeros only, not NaN. |
| **`array_has` with a zero.** `array_has(l, 0.0)` does not find `-0.0`. `list.contains(0.0)` is sent as `array_has(l, -0.0) OR array_has(l, 0.0)`. | [`_contains`, L579-584][pred-contains] | [lance#9316](https://github.com/lance-format/lance/issues/9316) (closed) | [lance#9628](https://github.com/lance-format/lance/pull/9628) | **pylance 15.0.0-beta.2**; no stable release yet | With `pylance>=15`, send `array_has(l, 0.0)`. Lance rewrites it to `array_has_any` over both encodings. That also covers NaN needles, so `list.contains(NaN)` can be pushed, which today declines. |
| **Float literal against an integer column.** Lance refuses `i > 1.5` ("could not convert to literal of type Int64"). The integer side is cast to `double`, which costs its scalar index. | [`_compare`, L402-408][pred-compare]; [`_is_in`, L513-514][pred-isin] | [lance#9317](https://github.com/lance-format/lance/issues/9317) (closed) | [lance#9333](https://github.com/lance-format/lance/pull/9333) | **pylance 15.0.0-beta.2**; no stable release yet | With `pylance>=15`, stop casting. `i > 1.5`, `i = 1.5`, `i BETWEEN 1 AND 2.5` and `i IN (1.5, 3.0)` are exact on 15.0.0b3 and use the index (`i > 1.5` plans as `[i > 1]`). Lance does not unwrap an explicit `CAST(i AS double)` (left for a follow-up in #9333), so the cast has to go for the index to come back. Only a bare column is refused: a computed integer such as `(i + 1) > 1.5` already plans on pylance 12. |
| **Float64 literal against a Float32 column.** Lance narrows a bare (untyped) float literal to a Float32 column, where Polars widens the column: `f > 0.1` and `f IN (0.1, 3.0)` return wrong rows. The column is cast to `double` instead (comparisons and float `is_in` lists), and an integer `is_in` list over a float column is cast to Float64 first. The cast costs the column's scalar index. | [`_compare`, L405-408][pred-compare]; [`_is_in`, L500 and L514][pred-isin] | [lance#9318](https://github.com/lance-format/lance/issues/9318) (closed) | [lance#9512](https://github.com/lance-format/lance/pull/9512) | **pylance 14.0.0** | Respell rather than drop. A bare literal is still narrowed on 15.0.0b3, so the column must not meet a bare literal. With `pylance>=14`, put the cast on the literal: `f > CAST(0.5 AS double)` is narrowed to `[f > 0.5]` on the index when the value is exact in Float32, and compared in double, unindexed but correct, when it is not (`CAST(0.1 AS double)`). `f IN (CAST(0.5 AS double), CAST(3.0 AS double))` uses the index too. On pylance 12 the typed literal is already correct, just unindexed. |
| **`xor`.** `flag != (id > 0)` fails to plan, because Lance converts every literal on the right of a boolean column to Boolean. `a ^ b` is expanded to `(a AND NOT b) OR (NOT a AND b)`, and declines unless both sides are exact. | [`_binary_predicate`, L375-387][pred-xor] | [lance#9319](https://github.com/lance-format/lance/issues/9319) (closed) | [lance#9322](https://github.com/lance-format/lance/pull/9322) | **pylance 14.0.0** | Keep the expansion. `a != b` plans on 14, but it never reaches a scalar index, while the expansion does: `(i > 2) ^ (i < 6)` becomes `OR([i >= 6], [i <= 2])` on the index. The exactness requirement holds for either spelling. |
| **`eq_missing` / `ne_missing`.** Lance rejects `IS [NOT] DISTINCT FROM`, so null-safe equality is only lowered against a non-null literal, as `(x = v) IS TRUE` / `IS NOT TRUE`; between two columns it declines. | [`_NULL_SAFE`, L53][pred-nullsafe-const]; [`_binary_predicate`, L367-373][pred-nullsafe] | none | [lance#9331](https://github.com/lance-format/lance/pull/9331) | **pylance 14.0.0** for non-float values; **15.0.0-beta.2** for floats | With `pylance>=14`, lower two non-float values as `x IS NOT DISTINCT FROM y`. Between two floats it is still wrong on 14 (`-0.0` vs `0.0`, NaN sign) and right on 15 ([lance#9628](https://github.com/lance-format/lance/pull/9628) covers both operators). Against a literal, `IS NOT DISTINCT FROM` uses no index, and neither does `IS TRUE`; an indexable spelling works on pylance 12 already, see below. |
| **`x ** 0`.** DataFusion's simplifier folds `power(x, 0)` to `1`, turning null rows into `1`. A zero or negative exponent declines. | [`_power`, L786][pred-power] | [datafusion#24246](https://github.com/apache/datafusion/issues/24246) | [datafusion#24247](https://github.com/apache/datafusion/pull/24247) | **DataFusion 55.0.0**; no Lance release yet (still on 54.1.0 through 15.0.0-beta.3; upgrade in [lance#9979](https://github.com/lance-format/lance/pull/9979), open) | Then allow exponent 0. A negative exponent keeps declining: `0 ** -1` fails the scan (see below). |
| **Float `//`.** DataFusion's one-argument `trunc` over an array maps `-0.0` to `0.0` (`if x == 0_f64 { 0_f64 } else { x.trunc() }` in `math/trunc.rs`), while its scalar path and `trunc(x, 0)` keep the sign. Float floor division declines. | [`_remainder_or_floor`, L683][pred-floordiv] (float operands fail the integer check) | [datafusion#25702](https://github.com/apache/datafusion/issues/25702) (closed) | [datafusion#25732](https://github.com/apache/datafusion/pull/25732) | Merged 2026-10-17, in no DataFusion release yet (not in 55.2.0, not backported to `branch-55`), so DataFusion 56 and a Lance upgrade to it | Does not need to wait: `` `floor`(a / b) `` works on pylance 12 and keeps `-0.0`, see below. `trunc(x, 0)` keeps `-0.0` on 12, 14 and 15.0.0b3 too, while `trunc(x)` loses it on all three. |
| **No `FLOOR` / `CEIL` in Lance filters.** Lance's planner has no branch for `sqlparser`'s `Floor` / `Ceil` forms ("is not supported SQL in lance"). Float `%` spells Polars' `a - b * floor(a / b)` as `trunc(q) - CAST(trunc(q) - q > 0.0 AS …)`. | [`_float_modulus`, L712][pred-floatmod] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | [lance#9584](https://github.com/lance-format/lance/pull/9584) (open, conflicting) | Not yet | Possible today with a backtick-quoted call, see below. |
| **No `CASE` in Lance filters.** Same planner gap. `when/then/otherwise` declines. | [`_predicate`, L339][pred-case] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | [lance#9584](https://github.com/lance-format/lance/pull/9584) (open, conflicting) | Not yet | `CASE` is a keyword, not a function, so there is no backtick spelling; this waits for #9584. |
| **No `SUBSTRING` in Lance filters.** Same planner gap, for both `SUBSTRING(…)` and `SUBSTR(…)`. `str.slice` declines (it has no branch in the lowering). | [`docs/PUSHDOWN.md`, L119][docs-slice] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | [lance#9584](https://github.com/lance-format/lance/pull/9584) (open, conflicting) | Not yet | Possible today with a backtick-quoted call, see below. |

## Possible without an upstream fix

These were found while checking the rows above. Each works on pylance 12.0.0, 14.0.0 and
15.0.0b3 and matched Polars on the probed values.

- **Backtick-quoted `floor`, `ceil` and `substr`.** `sqlparser` turns `FLOOR(x)`,
  `CEIL(x)` and `SUBSTRING(…)` into special syntax nodes that Lance's planner
  rejects. Quoting the name makes it an ordinary function call, which resolves to
  DataFusion's function:
    - `` `floor`(x) `` and `` `ceil`(x) `` keep `-0.0` and work on Float32. DataFusion
      even turns `` `floor`(x) = 1.0 `` into the index range `[x >= 1 && x < 2]`.
    - Float `%` can be spelled ``a - b * `floor`(a / b)`` directly instead of the
      `trunc` correction, and float `//` as ``` `floor`(a / b) ``` (the NaN-sign
      correction still applies to comparisons on the result below pylance 15).
    - `` `substr`(s, start, length) `` counts characters, as `str.slice` does (checked
      with multi-byte UTF-8), so `str.slice` with a non-negative offset can be pushed.
      A negative offset counts from the end in Polars and not in SQL, and stays
      declined.
- **Indexable null-safe equality against a literal.** `(x = v AND x IS NOT NULL)` for
  `eq_missing` and `(x != v OR x IS NULL)` for `ne_missing` are two-valued like
  `IS TRUE` / `IS NOT TRUE`, also under `NOT`, but reach the scalar index
  (`[i = 3]@i_idx`, and `OR(NOT([i = 3]), [i IS NULL])` for the negation), which
  `(x = v) IS TRUE` does not.
- **`eq_missing(None)` / `ne_missing(None)`** are `x IS NULL` / `x IS NOT NULL`. They
  decline today because the literal must be non-null ([`_is_non_null_literal`][pred-nonnull]).
- **NaN literals, from pylance 15.** `x = NaN`, `x > NaN`, `x < NaN` and `x >= NaN`
  (as `CAST('NaN' AS double)`) match Polars on 15.0.0b3 and not before. NaN and
  infinite literals decline in [`_scalar`][pred-scalar] today.

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
- A bare `NULL` filter is rejected ("does not return a boolean"), hence
  `CAST(NULL AS boolean)`; `IN ()` is a parse error in `sqlparser`. Both still hold on
  15.0.0b3.
- A double-quoted identifier is not rejected: `"i"` parses as the string `'i'`, so
  `"i" = 3` fails to cast and `"s" = 's'` would be a constant comparison. Identifiers
  are backtick-quoted ([`_quote`][pred-quote]).
- Unclear, not investigated: `ln` of a Float32 and `log10` are an ulp off.

[pred-nan]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L1178-L1191
[pred-zeros]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L1194-L1197
[pred-contains]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L579-L584
[pred-compare]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L402-L408
[pred-isin]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L498-L514
[pred-xor]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L375-L387
[pred-nullsafe-const]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L53
[pred-nullsafe]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L367-L373
[pred-nonnull]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L1208-L1212
[pred-scalar]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L1088-L1090
[pred-quote]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L1012-L1016
[pred-power]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L773-L791
[pred-floordiv]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L664-L692
[pred-floatmod]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L694-L714
[pred-case]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/src/polars_pylance/_predicate.py#L339-L342
[docs-slice]: https://github.com/jonasdedden/polars-pylance/blob/2145054690d3c674c488b6aa0efe44415bda0897/docs/PUSHDOWN.md?plain=1#L119
