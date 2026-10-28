# Upstream fixes that would remove polars-pylance workarounds

Status as of 2026-10-28. polars-pylance requires `pylance>=12`;
[#63](https://github.com/jonasdedden/polars-pylance/pull/63) raises that to `pylance>=14`.
The latest stable pylance is [14.0.1](https://github.com/lance-format/lance/releases/tag/v14.0.1)
(2026-10-28, an `object_store` fix over 14.0.0); 15.0.0 is in beta (`15.0.0b4`, from
`https://pypi.fury.io/lance-format/`). Every Lance release up to 15.0.0-beta.4 builds on
DataFusion 54.1.0. Lance `main` moved to DataFusion 55.2.0 on 2026-10-28 with
[lance#10397](https://github.com/lance-format/lance/pull/10397), which superseded the closed
[lance#9979](https://github.com/lance-format/lance/pull/9979). The 15.0.0 betas are cut from
`main`, so the next one will likely carry DataFusion 55. Code links point at `main` @
`2145054`.

Every row below was checked by running both the SQL the lowering emits and the naive
SQL it avoids against pylance 12.0.0, 14.0.0, 14.0.1, 15.0.0b3 and 15.0.0b4. The data
holds NaN of both signs, signed zeros, infinities, integers past 2^53 and multi-byte
strings, split into fragments so that some zones hold only NaN. Each SQL spelling was
compared with Polars row by row, plain and under `NOT`, without an index and with
BTREE, ZONEMAP and BITMAP indices, and `explain_plan()` was read for scalar index use.
14.0.1 behaves as 14.0.0, and 15.0.0b4 as 15.0.0b3, in every probe. The test suite
passes unchanged on all five versions.

## What each pylance floor unlocks

| Floor | Change in the lowering |
| --- | --- |
| none (works on pylance 12) | Done in #63: float `%` via a backtick-quoted `floor`, `eq_missing` / `ne_missing` against a literal on the index, and NaN guards that a zone map cannot drop. Still open: float `//` via `floor`, `str.slice` via `substr` (any offset), `when/then/otherwise` without `CASE`, and `eq_missing(None)` on the direct and prefilter paths. See [Possible without an upstream fix](#possible-without-an-upstream-fix). |
| `pylance>=14` (#63) | A Float64 literal against a Float32 column can keep the scalar index (cast the literal, not the column). On the `scan_lance(...).filter(...)` path Polars' optimizer has already wrapped the column in `cast(Float64, strict=False)`, so the lowering would have to strip that cast to benefit. `eq_missing` / `ne_missing` between two non-float values. Nothing becomes removable: `xor` plans as `!=`, but only the expansion reaches an index. |
| `pylance>=15` | Drop the NaN-sign and signed-zero corrections, the zone-map terms from #63 and the zero `array_has` disjunction, and with them the `nan_free` tracking only `min_horizontal` / `max_horizontal` would still read. Prototyped against `2145054`, that removes about 85 lines of `_predicate.py` net; every row-by-row test passes on 15.0.0b3/b4 and 97 fail on 12 and 14. Compare integer columns with float literals directly where the literal is below 2^53 in magnitude. `eq_missing` / `ne_missing` between two float values. NaN literals and `list.contains(NaN)` can be pushed. |
| Lance on DataFusion 55 | `x ** 0`. In Lance `main`, not yet released. |
| Lance on DataFusion 56 | One-argument `trunc` keeps `-0.0`. Not needed if float `//` uses `floor`. |
| [lance#9584](https://github.com/lance-format/lance/pull/9584) merged | `CASE`, `FLOOR`, `CEIL` and `SUBSTRING` spelled directly. Cosmetic: each has a working spelling today. |

## Workarounds

| Workaround | Code in polars-pylance | Upstream issue | Upstream PR | Fixed in | Then |
| --- | --- | --- | --- | --- | --- |
| **Zero literals.** Lance compared `-0.0` and `0.0` unequal against a zero literal, so every float `=`, ordering, `IN` and `BETWEEN` against zero spelled out both zeros, and the integer `%` / `//` sign correction used a long `(r < 0 AND b > 0) OR …` form. | Removed in [#41](https://github.com/jonasdedden/polars-pylance/pull/41). | none | [lance#6236](https://github.com/lance-format/lance/pull/6236) | **pylance 12.0.0**, already required | Done. |
| **NaN sign.** Lance orders a NaN with its sign bit set (what `0 / 0` gives on x86) below `-inf`; Polars puts every NaN above every number. Against a literal, `x < -inf` is added to or excluded from ordering comparisons; between two float values, `nanvl(x, NaN)` gives every NaN the positive sign. | [`_float_comparison`, L1178-1191][pred-nan]; respelled in #63 (next row) | [lance#9315](https://github.com/lance-format/lance/issues/9315) (closed) | [lance#9628](https://github.com/lance-format/lance/pull/9628), which superseded [lance#9324](https://github.com/lance-format/lance/pull/9324) (closed unmerged) | **pylance 15.0.0-beta.2**; no stable release yet | With `pylance>=15`, send the plain comparison. Lance now adds `x < -inf` itself (`x > 1.0` plans as `OR([x > 1], [x < -inf])` on the index), and compares two values ignoring the NaN sign, computed ones and Float32 included. Until then the guard stays and is harmless on 15: Lance rewrites the inner `x < -inf` into an empty range and adds its own. |
| **Zone maps skip NaN-only zones.** Below pylance 15, a ZONEMAP index treats a zone holding only positive NaNs as empty, so `x > c`, and `NOT (x < c AND x >= -inf)`, lose its rows even with the NaN-sign guard. | Not covered at `2145054`. [#63](https://github.com/jonasdedden/polars-pylance/pull/63) adds `OR x = CAST('NaN' AS double)` to `>` / `>=` and spells `<` / `<=` as `AND NOT isnan(x)`; both still use a BTREE index. | none | [lance#10102](https://github.com/lance-format/lance/pull/10102) | **pylance 15.0.0-beta.1**, also for zone maps written by 12 or 14 | With `pylance>=15`, drop it together with the NaN-sign guard. |
| **Signed zeros between two float values.** Column-to-column comparisons use Arrow's total order, so `-0.0 = 0.0` is false. The zero pair is decided explicitly: `… OR (x = 0 AND y = 0)` / `… AND NOT (…)`. | [`_float_comparison`, L1194-1197][pred-zeros] | [lance#9316](https://github.com/lance-format/lance/issues/9316) (closed) | [lance#9628](https://github.com/lance-format/lance/pull/9628), which superseded [lance#9323](https://github.com/lance-format/lance/pull/9323) (closed unmerged) | **pylance 15.0.0-beta.2**; no stable release yet | With `pylance>=15`, drop the zero pair together with `nanvl`: all six operators match Polars on 15.0.0b3/b4, also for computed values and Float32 against Float64. [datafusion#22835](https://github.com/apache/datafusion/pull/22835) (DataFusion 55) does not matter here. Lance compares through its own `_lance_compare_floats` UDF, and #22835 normalizes zeros only, not NaN. |
| **`array_has` with a zero.** `array_has(l, 0.0)` does not find `-0.0`. `list.contains(0.0)` is sent as `array_has(l, -0.0) OR array_has(l, 0.0)`. | [`_contains`, L579-584][pred-contains] | [lance#9316](https://github.com/lance-format/lance/issues/9316) (closed) | [lance#9628](https://github.com/lance-format/lance/pull/9628) | **pylance 15.0.0-beta.2**; no stable release yet | With `pylance>=15`, send `array_has(l, 0.0)`. Lance rewrites it to `array_has_any` over both encodings, for Float64 and Float32 lists. That also covers NaN needles, so `list.contains(NaN)` can be pushed, which today declines. |
| **Float literal against an integer column.** Lance refuses `i > 1.5` ("could not convert to literal of type Int64"). The integer side is cast to `double`, which costs its scalar index. | [`_compare`, L402-408][pred-compare]; [`_is_in`, L513-514][pred-isin] | [lance#9317](https://github.com/lance-format/lance/issues/9317) (closed) | [lance#9333](https://github.com/lance-format/lance/pull/9333) | **pylance 15.0.0-beta.2**; no stable release yet | Not simply "stop casting". #9333 compares the integer exactly (`i > 1.5` plans as `[i > 1]`), while Polars converts it to Float64 first. They agree below 2^53: on 15.0.0b4, `i > 9007199254740992.0`, `i = 2^53`, `i < -2^53` and `i IN (2^53)` return different rows for `i = 2^53 + 1`. So the cast may only go for a literal below 2^53 in magnitude, or a column of at most 32 bits. Polars' optimizer also wraps the column itself (`TRY_CAST(i AS double)` reaches the scan), and Lance does not unwrap an explicit cast, so the lowering would have to strip it. The upstream change that removes all of this: Lance turning `CAST(int AS double) op literal` (and `TRY_CAST`) into an index range when the literal is below 2^53, the follow-up #9333 left open. Only a bare column is refused: a computed integer such as `(i + 1) > 1.5` already plans on pylance 12. |
| **Float64 literal against a Float32 column.** Lance narrows a bare (untyped) float literal to a Float32 column, where Polars widens the column: `f > 0.1` and `f IN (0.1, 3.0)` return wrong rows. The column is cast to `double` instead (comparisons and float `is_in` lists), and an integer `is_in` list over a float column is cast to Float64 first. The cast costs the column's scalar index. | [`_compare`, L405-408][pred-compare]; [`_is_in`, L500 and L514][pred-isin] | [lance#9318](https://github.com/lance-format/lance/issues/9318) (closed) | [lance#9512](https://github.com/lance-format/lance/pull/9512) | **pylance 14.0.0** | Respell rather than drop. A bare literal is still narrowed on 15.0.0b4, so the column must not meet a bare literal. With `pylance>=14`, put the cast on the literal: `f > CAST(0.5 AS double)` is narrowed to `[f > 0.5]` on the index when the value is exact in Float32, and compared in double, unindexed but correct, when it is not (`CAST(0.1 AS double)`). `f IN (CAST(0.5 AS double), CAST(3.0 AS double))` uses the index too. On the `scan_lance(...).filter(...)` path the column arrives as `TRY_CAST(f AS double)` from Polars' optimizer, which the lowering would have to strip; that is lossless, so it is sound. Upstream, applying #9512's "narrow only when exact" to bare literals too would let the plain spelling stand. |
| **`xor`.** `flag != (id > 0)` fails to plan, because Lance converts every literal on the right of a boolean column to Boolean. `a ^ b` is expanded to `(a AND NOT b) OR (NOT a AND b)`, and declines unless both sides are exact. | [`_binary_predicate`, L375-387][pred-xor]; comment updated in #63 | [lance#9319](https://github.com/lance-format/lance/issues/9319) (closed) | [lance#9322](https://github.com/lance-format/lance/pull/9322) | **pylance 14.0.0** | Keep the expansion. `a != b` plans on 14, but it never reaches a scalar index, while the expansion does: `(i > 2) ^ (i < 6)` becomes `OR([i >= 6], [i <= 2])` on the index. The exactness requirement holds for either spelling. To drop it, Lance's index planner would have to expand `bool != bool` itself. |
| **`eq_missing` / `ne_missing`.** Lance rejects `IS [NOT] DISTINCT FROM`, so null-safe equality is only lowered against a non-null literal, as `(x = v) IS TRUE` / `IS NOT TRUE`; between two columns it declines. | [`_NULL_SAFE`, L53][pred-nullsafe-const]; [`_binary_predicate`, L367-373][pred-nullsafe]; respelled in #63 | none | [lance#9331](https://github.com/lance-format/lance/pull/9331) | **pylance 14.0.0** for non-float values; **15.0.0-beta.2** for floats | Against a literal, #63 spells them `(x = v AND x IS NOT NULL)` / `(NOT (x = v) OR x IS NULL)`, which reach the index; `IS TRUE` and `IS NOT DISTINCT FROM` do not. With `pylance>=14`, two non-float values could be lowered as `x IS NOT DISTINCT FROM y`. Between two floats it is still wrong on 14 (`-0.0` vs `0.0`, NaN sign) and right on 15 ([lance#9628](https://github.com/lance-format/lance/pull/9628) covers both operators). `eq_missing(None)` already reaches the scan as `IS NULL` through Polars' optimizer; only direct calls and prefilters decline it. |
| **`x ** 0`.** DataFusion's simplifier folds `power(x, 0)` to `1`, turning null rows into `1`. A zero or negative exponent declines. | [`_power`, L786][pred-power] | [datafusion#24246](https://github.com/apache/datafusion/issues/24246) | [datafusion#24247](https://github.com/apache/datafusion/pull/24247) | **DataFusion 55.0.0**, which now folds only a non-nullable base. In Lance `main` since [lance#10397](https://github.com/lance-format/lance/pull/10397); in no release yet (still 54.1.0 through 15.0.0-beta.4) | Then allow exponent 0. A negative exponent keeps declining: Arrow fails the scan on `0 ** -1` ("zero raised to a negative power is undefined"), even for a float base, where IEEE `pow` gives `inf`. The other `**` declines in #63 are not upstream bugs, see below. |
| **Float `//`.** DataFusion's one-argument `trunc` over an array maps `-0.0` to `0.0` (`if x == 0_f64 { 0_f64 } else { x.trunc() }` in `math/trunc.rs`), while its scalar path and `trunc(x, 0)` keep the sign. Float floor division declines. | [`_remainder_or_floor`, L683][pred-floordiv] (float operands fail the integer check) | [datafusion#25702](https://github.com/apache/datafusion/issues/25702) (closed) | [datafusion#25732](https://github.com/apache/datafusion/pull/25732) | Merged 2026-10-17, in no DataFusion release yet (still unfixed in 55.2.0, not backported to `branch-55`), so DataFusion 56 and a Lance upgrade to it | Does not need to wait: `` `floor`(a / b) `` works on pylance 12 and matches Polars' column-by-column `//` bit for bit, `-0.0`, NaN and Float32 included. A fractional float literal divisor must still decline, as for `%`: Polars' column-by-literal kernel gives `0.3 // 0.1 = 3.0` where `floor(0.3 / 0.1)` is `2.0` (on a one-row frame, Polars itself gives `2.0`). `trunc(x, 0)` keeps `-0.0` on every probed version, while `trunc(x)` loses it on all of them. |
| **No `FLOOR` / `CEIL` in Lance filters.** Lance's planner has no branch for `sqlparser`'s `Floor` / `Ceil` forms ("is not supported SQL in lance"). Float `%` spells Polars' `a - b * floor(a / b)` as `trunc(q) - CAST(trunc(q) - q > 0.0 AS …)`. | [`_float_modulus`, L712][pred-floatmod] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | [lance#9584](https://github.com/lance-format/lance/pull/9584) (open, conflicting) | Not yet | Not needed: #63 spells float `%` as ``a - b * `floor`(a / b)``, which matches Polars bit for bit without the `+ 0.0`. |
| **No `CASE` in Lance filters.** Same planner gap. `when/then/otherwise` declines. | [`_predicate`, L339][pred-case] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | [lance#9584](https://github.com/lance-format/lance/pull/9584) (open, conflicting) | Not yet | `CASE` is a keyword, so there is no backtick spelling, but it does not need to wait for #9584, see below. |
| **No `SUBSTRING` in Lance filters.** Same planner gap, for both `SUBSTRING(…)` and `SUBSTR(…)`. `str.slice` declines (it has no branch in the lowering). | [`docs/PUSHDOWN.md`, L119][docs-slice] | [lance#9560](https://github.com/lance-format/lance/issues/9560) | [lance#9584](https://github.com/lance-format/lance/pull/9584) (open, conflicting) | Not yet | Possible today with a backtick-quoted call, see below. |

## Possible without an upstream fix

These were found while checking the rows above. Each works on pylance 12.0.0, 14.0.x and
15.0.0b3/b4 and matched Polars row by row, plain and under `NOT`.

- **Backtick-quoted `floor`, `ceil`, `substr` and `substring`.** `sqlparser` turns
  `FLOOR(x)`, `CEIL(x)` and `SUBSTRING(…)` into special syntax nodes that Lance's
  planner rejects. Quoting the name makes it an ordinary function call, which resolves
  to DataFusion's function:
    - `` `floor`(x) `` and `` `ceil`(x) `` keep `-0.0` and Float32. DataFusion even turns
      `` `floor`(x) = 1.0 `` into the index range `[x >= 1 && x < 2]`.
    - Float `%` as ``a - b * `floor`(a / b)`` is done in #63. Float `//` as
      ``` `floor`(a / b) ``` needs the fractional-literal decline above, and below
      pylance 15 the NaN-sign correction on comparisons of the result.
    - `` `substr`(s, start, length) `` counts characters, as `str.slice` does (checked
      with multi-byte UTF-8). A non-negative offset is `` `substr`(s, offset + 1, length) ``.
      A negative one counts from the end in Polars, and
      `` `substr`(s, character_length(s) + offset + 1, length) `` matches it too,
      offsets past the start included, because both clip the requested window to the
      string.
- **`when/then/otherwise` without `CASE`.** As a predicate,
  `((c) IS TRUE AND a) OR ((c) IS NOT TRUE AND b)` matches Polars, which takes the
  `otherwise` branch for a null condition, also under `NOT`. As a value,
  `nvl2(NULLIF(CAST((c) IS TRUE AS bigint), 0), a, b)` works for numbers and strings,
  and is lazy like `CASE` (DataFusion rewrites `nvl2` into one), so `6 / i` with
  `i = 0` in the untaken branch does not fail the scan. A Boolean first argument
  fails `nvl2`'s type coercion, hence the `bigint`.
- **`eq_missing(None)` / `ne_missing(None)`** are `x IS NULL` / `x IS NOT NULL`. Through
  Polars' optimizer they already arrive that way; direct calls and prefilters decline
  them because the literal must be non-null ([`_is_non_null_literal`][pred-nonnull]).
- **NaN literals, from pylance 15.** `=`, `!=`, `<`, `>=` and `IN` against
  `CAST('NaN' AS double)` match Polars on 15 and not before (`>` and `<=` already do
  on 12, without a zone map). NaN and infinite literals decline in
  [`_scalar`][pred-scalar] today.

## Not upstream bugs

These declines follow from Lance/DataFusion behaving as SQL or IEEE 754 specify, or
from deliberate choices, so there is nothing to file:

- `round` breaks ties away from zero (SQL), Polars to even.
- `power` is `powf` in Float64. Polars (`pow_on_floats`) raises to an integer exponent
  in wrapping integer arithmetic and a Float32 in Float32, takes `sqrt` for 0.5 (so
  `(-inf) ** 0.5` is NaN there and `inf` in Lance), and multiplies out whole exponents
  2 to 9, an ulp off `powf` on up to half the rows. At `2145054` the lowering pushed
  several of these: `(f32 ** 3) < f32 * f32 * f32` and `(i32 ** 2) > 0` returned
  wrong rows. #63 lowers only a Float64 exponent, of an integer or Float64 base, and
  outside those cases, where both sides match bit for bit.
- `0 ** -1` and `i64::MIN // -1` fail the scan (as in Postgres); Polars gives `inf` / wraps.
- A float cast to string prints `1e16`, Polars `1e+16`.
- `utf8 + utf8` is rejected; text concatenation is `||`.
- UInt64 with a signed integer is coerced to `Decimal128(20, 0)`
  ([datafusion#14223](https://github.com/apache/datafusion/pull/14223)), so their
  quotient is a `Decimal128(24, 4)` with a fraction: `u / NULLIF(3, 0)` gives `3.3333`
  for 10, and even `(u / 3) = 3` inside a filter divides as a decimal. Only a bare
  top-level `u / 3`, whose literal Lance coerces to the column's type, divides as
  integers. That, not the `bigint` correction term, is why UInt64 `//` declines.
- `if()` and `mod()` are not DataFusion functions; `%` works.
- A bare `NULL` filter is rejected ("does not return a boolean"), hence
  `CAST(NULL AS boolean)`; `IN ()` is a parse error in `sqlparser`. Both still hold on
  15.0.0b4.
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
