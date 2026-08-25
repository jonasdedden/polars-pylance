# Benchmark

What happens between **1 GiB and 200 GiB**, where the two implementations stop
behaving alike — one streams and the other materialises, and below a few GiB you
cannot tell them apart.

It compares polars-pylance against
[`polars-lance`](https://pypi.org/project/polars-lance/) (a compiled Rust
extension solving the same problem) across a nine-tier size ladder, measuring
wall time and peak RSS for every case, plus whether the query completes at all
under a fixed memory budget.

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

The instance is **not** destroyed automatically — the datasets take a while to
regenerate, so you probably want it alive for follow-ups:

```sh
pulumi destroy --yes
```

The instance store is ephemeral: stopping the instance destroys the data, so
"stop to save money" does not apply. It is running or it is gone.

### Cost

`m8id.4xlarge` is about $1.26/hr in eu-central-1. A full nine-tier run takes
roughly 1-1.5 h including data generation, so about **$1.50-2.00**.

## Running it anywhere else

The scripts have no AWS dependency — `infra/` is only there to obtain a
suitable machine. On any box with a fast local filesystem:

```sh
export BENCH_ROOT=/mnt/fast-nvme
uv run bench/gen.py 2000000 4000000 8000000
BENCH_PYTHON=$(which python) uv run bench/run_matrix.py 2000000,4000000,8000000 55 8
uv run bench/analyse.py "$BENCH_ROOT/results.jsonl"
```

`run_matrix.py` takes `<ladder> <generous-cap-GiB> <fixed-budget-GiB>`. It uses
`systemd-run --scope` for the memory caps, so it needs systemd with cgroup v2;
without it, drop the caps and only the scaling numbers are meaningful.

## Plots

```sh
uv run --group bench bench/plot.py bench/results-m8id4xl.jsonl \
    --out bench/plots
```

Three interactive pages in `plots/`:

- `scaling.html` and `fixed-budget.html` — one panel per case, with **both costs
  on the same axes**: runtime on the left (solid, circles), peak memory on the
  right (dotted, diamonds), colour per implementation. They belong together
  because the interesting cases are the ones where the two metrics disagree —
  the full scan is 1.2× slower *and* 2.5× lighter, and separate pages hide that.
- `ratios.html` — polars-pylance ÷ polars-lance, where below 1.0 means
  polars-pylance wins.

Every one of the 212 datapoints is on them, failures included: OOM kills and
thread panics are drawn as distinct markers on the runtime axis rather than left
as gaps, so "did not finish" never looks like "was not measured".

Pages link plotly from the CDN by default; `--offline` writes a sibling
`plotly.min.js` instead.

`--static` additionally writes PNG and SVG of three selected panels into
`plots/static/` (needs `kaleido`); those are the ones embedded in
[`COMPARISON.md`](../COMPARISON.md).

### Viewing the interactive pages

GitHub serves committed `.html` as source, not as a rendered page, so clicking
one here shows markup. Two ways to actually look at them:

- **Download and open locally** — `curl -LO` the raw file, or clone the repo and
  open `bench/plots/scaling.html`. The CDN build needs a network; use
  `--offline` if you want it self-contained.
- **Proxy it** — paste the file's GitHub URL into
  [htmlpreview.github.io](https://htmlpreview.github.io/), e.g.
  `https://htmlpreview.github.io/?https://github.com/jonasdedden/polars-lance/blob/main/bench/plots/scaling.html`

The static PNGs in `COMPARISON.md` exist precisely because neither of those is a
click, and the headline results should not need one.

## Layout

| file | role |
| --- | --- |
| `gen.py` | writes the size ladder; payload bytes are random so nothing compresses away |
| `cases.py` | one measurement per process — peak RSS is a per-process high-water mark, so a materialising run would poison every later reading in the same interpreter |
| `run_matrix.py` | drives the matrix; each measurement gets its own cgroup scope with swap disabled, so an over-budget run is killed cleanly instead of swapping |
| `analyse.py` | renders `results.jsonl` as per-case scaling tables |
| `plot.py` | renders `results.jsonl` as interactive Plotly pages |
| `infra/` | Pulumi program, SSM wrapper, end-to-end driver |

## What it measures

Seven read shapes (full scan, projection-only, selective filter, half-selectivity
filter with payload, string predicate, `head()` limit pushdown, top-k sort), the
write path, and four cases only polars-pylance supports (fragment-parallel write,
pinned version, `_rowid`, sharded fragment scans).

The matrix runs two passes, which answer different questions:

- **`scaling`** — a generous cap (55 GiB of 64), high enough to be a safety net
  rather than a constraint. Shows how peak RSS and runtime grow with dataset
  size: is memory flat or does it track the data?
- **`fixed-budget`** — memory pinned (8 GiB) with swap disabled, then the data
  grown past it. Only the full scan and the write, because this is the question
  that matters in production: not which is 30% faster, but which one still
  finishes.

Put another way: `scaling` measures *how much* memory a job needs;
`fixed-budget` measures *whether the job runs* when memory is what you have.

Both use `systemd-run --scope` with `MemoryMax` and `MemorySwapMax=0`, so
exceeding the budget is a clean kill rather than a swap death-spiral.

Failures are recorded by kind. `polars-lance` has a thread-safety bug where its
scanner is not `Send` and a scan can abort with `PyLanceScanner is unsendable`;
the driver retries three times and records `panic-unsendable` separately from
`OOM-killed`, so a crash is never miscounted as a memory limit.
