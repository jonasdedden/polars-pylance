# Benchmark

What happens between **1 GiB and 200 GiB**, where the two implementations stop
behaving alike: one streams and the other materialises, and below a few GiB you
cannot tell them apart.

It compares polars-pylance against
[`polars-lance`](https://pypi.org/project/polars-lance/) (a compiled Rust
extension solving the same problem) across a nine-tier size ladder, measuring
wall time and peak RSS for every case, plus whether the query completes at all
under a fixed memory budget, and what a pushed-down predicate can reach once
scalar indices exist.

## Why it needs a real machine

Two constraints make a laptop the wrong host:

- **Local NVMe is required.** EBS or any network volume measures the network,
  not the format. The Pulumi program picks an instance type with an instance
  store and mounts it; nothing touches the root volume.
- **`/tmp` on many Linux desktops is a RAM disk**, which silently consumes the
  memory being measured. `BENCH_ROOT` must point at real storage.

## Running it on AWS

```sh
cd bench/infra
pulumi stack init bench
pulumi config set aws:region eu-central-1
pulumi config set instanceType m8id.4xlarge   # 16 vCPU, 64 GiB, 950 GB NVMe
./run_remote.sh --deploy
```

That provisions a dedicated VPC with **no inbound rules** (access is over SSM
Session Manager, so no SSH keys and no open ports), builds the wheel, uploads
it, generates the ladder, runs the matrix, and prints the tables.

The instance is **not** destroyed automatically. The datasets take a while to
regenerate, so you probably want it alive for follow-ups:

```sh
pulumi destroy --yes
```

The instance store is ephemeral: stopping the instance destroys the data, so
"stop to save money" does not apply. It is running or it is gone.

### Cost

`m8id.4xlarge` is about $1.26/hr in eu-central-1. A full nine-tier run takes
roughly 1.5-2 h including data generation and index builds, so about
**$2.00-2.50**.

## Running it anywhere else

The scripts have no AWS dependency; `infra/` is only there to obtain a
suitable machine. On any box with a fast local filesystem:

```sh
export BENCH_ROOT=/mnt/fast-nvme
uv run bench/gen.py 2000000 4000000 8000000
BENCH_PYTHON=$(which python) uv run bench/run_matrix.py 2000000,4000000,8000000 55 8
uv run bench/analyse.py "$BENCH_ROOT/results.jsonl"
```

`run_matrix.py` takes `<ladder> <generous-cap-GiB> <fixed-budget-GiB>`. It uses
`systemd-run --scope` for the memory caps, so it needs systemd with cgroup v2,
and on a desktop that prompts for authorisation. `BENCH_CAPS=0` runs each
measurement directly instead: the scaling and indexed passes still mean what
they say, and the fixed-budget pass, which is nothing but a cap, is skipped.

The schema gained `text` and `ts` columns for the predicate cases, so a data
directory written before that has to be regenerated. `gen.py` only skips a tier
whose row count already matches, not its schema, so delete the old ladder.

## Plots

```sh
uv run --group bench bench/plot.py bench/results-m8id4xl.jsonl \
    --out bench/plots
```

Three interactive pages in `plots/`:

- `scaling.html`, `fixed-budget.html` and `indexed.html` have one panel per
  case, with **both costs on the same axes**: runtime on the left (solid, circles), peak memory on the
  right (dotted, diamonds), colour per implementation. They belong together
  because the interesting cases are the ones where the two metrics disagree:
  the full scan is 1.2× slower *and* 2.5× lighter, and separate pages hide that.
- `ratios.html` is polars-pylance ÷ polars-lance, where below 1.0 means
  polars-pylance wins.

Every datapoint is on them, failures included: OOM kills, thread panics and
unreadable expressions are drawn as distinct markers on the runtime axis rather
than left as gaps, so "did not finish" never looks like "was not measured".

Pages link plotly from the CDN by default; `--offline` writes a sibling
`plotly.min.js` instead.

`--static` additionally writes the selected panels into `plots/static/` (needs
`kaleido`); those are the ones embedded in
[`COMPARISON.md`](../COMPARISON.md). `--format` picks the image format and
defaults to `svg`, which is what is committed; `--format png` writes a 2x raster
instead, for somewhere that will not take vector.

### Viewing the interactive pages

GitHub serves committed `.html` as source, not as a rendered page, so clicking
one here shows markup. Two ways to actually look at them:

- **Download and open locally.** `curl -LO` the raw file, or clone the repo and
  open `bench/plots/scaling.html`. The CDN build needs a network; use
  `--offline` if you want it self-contained.
- **Proxy it.** Paste the file's GitHub URL into
  [htmlpreview.github.io](https://htmlpreview.github.io/), e.g.
  `https://htmlpreview.github.io/?https://github.com/jonasdedden/polars-lance/blob/main/bench/plots/scaling.html`

The static images in `COMPARISON.md` exist precisely because neither of those is
a click, and the headline results should not need one.

## Layout

| file | role |
| --- | --- |
| `gen.py` | writes the size ladder; payload bytes are random so nothing compresses away |
| `index.py` | builds the scalar indices the indexed pass needs, in place |
| `cases.py` | one measurement per process, because peak RSS is a per-process high-water mark and a materialising run would poison every later reading in the same interpreter |
| `run_matrix.py` | drives the matrix; each measurement gets its own cgroup scope with swap disabled, so an over-budget run is killed cleanly instead of swapping |
| `analyse.py` | renders `results.jsonl` as per-case scaling tables |
| `plot.py` | renders `results.jsonl` as interactive Plotly pages |
| `infra/` | Pulumi program, SSM wrapper, end-to-end driver |

## What it measures

Eleven read shapes, the write path, and four cases only polars-pylance supports
(fragment-parallel write, pinned version, `_rowid`, sharded fragment scans).

Four of the read shapes are about **predicate pushdown**, and they all put the
filter in front of the 512-byte payload column, which is where skipping rows is
worth something:

| case | predicate |
| --- | --- |
| `r_is_in` | `id.is_in(200 ids)` |
| `r_str` | `text.str.contains("-rare")`, one row in 10,000 |
| `r_arith` | `val * 2 > 1.999` |
| `r_temporal` | `ts.dt.hour() < 1` |

None of them can be expressed as a PyArrow expression, which is the limit of
what `scan_pyarrow_dataset` and `scan_delta` can offer Lance. polars-pylance
translates the Polars expression into a Lance SQL filter instead, so the
scanner does the work.

polars-lance pushes no predicate into Lance at all, which is an acknowledged
`TODO` on its side, so for these cases it reads the payload column in full and
filters afterwards. Measured: with and without Polars' predicate pushdown its
runtime is the same to within noise.

### The three passes

- **`scaling`** is a generous cap (55 GiB of 64), high enough to be a safety net
  rather than a constraint. Shows how peak RSS and runtime grow with dataset
  size: is memory flat or does it track the data?
- **`fixed-budget`** pins memory (8 GiB) with swap disabled and grows the data
  past it. Only the full scan and the write, because this is the question that
  matters in production: not which is 30% faster, but which one still finishes.
- **`indexed`** runs last, because `index.py` changes the datasets in place. It
  builds BTREE indices on `id` and `val`, BITMAP on `cat` and NGRAM on `text`,
  then re-runs the predicate cases. An index is only reachable by a reader that
  pushes the predicate down, so this is the same query and the same data with
  one implementation able to act on it.

`r_temporal` is the control in that pass: no index can answer `dt.hour()`, so it
should not move. `r_cat` is the counter-example, and it is in there on purpose:
a BITMAP index on four distinct values, asked for a predicate that keeps a
quarter of the table, is slower than the scan it replaces. `r_cat_noindex` is
the same query with `LanceScanOptions(use_scalar_index=False)`, which is the fix.

The first two passes use `systemd-run --scope` with `MemoryMax` and
`MemorySwapMax=0`, so exceeding the budget is a clean kill rather than a swap
death-spiral.

### Failures, and answers that disagree

Failures are recorded by kind, never as a gap:

- `OOM-killed` is the memory cap doing its job.
- `panic-unsendable` is a polars-lance thread-safety bug: its scanner is not
  `Send` and a scan can abort with `PyLanceScanner is unsendable`. The driver
  retries three times, because it is probabilistic.
- `expr-unsupported` is a polars-lance version mismatch. It links its own polars
  crate and re-decodes the expression Polars already handed it as a `pl.Expr`,
  so one built by a newer polars either fails to decode or decodes as a
  different variant. On 0.5.0 against polars 1.44, `is_in`, `is_between`,
  `is_null`, `abs`, `starts_with`, `dt.hour` and `is_nan` fail to decode, and
  `str.contains` arrives as `is_leap_year`.

  The query itself is fine: the error escapes the scan node instead of being
  reported as "predicate not applied", which is the path `register_io_source`
  already provides and which Polars uses when its own decode fails. So the read
  cases retry once with `predicate_pushdown=False`, which runs, and record
  `"pushdown": "declined-by-scan"`. That is the same work, since the predicate
  was never going into Lance, so the comparison stays like for like and the
  tables mark it with a `+`. `expr-unsupported` remains as the status for
  anything the retry does not cover.

`analyse.py` also compares the two answers and prints `DISAGREE` instead of a
speed ratio when they differ, with a tolerance on float sums, whose aggregation
order depends on batch size. That is not hypothetical: on 0.5.0,
`-pl.col("id") < -x` and `fill_null` both return zero rows where the correct
answer is not zero.
