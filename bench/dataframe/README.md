# Dataframe benchmarks

This directory compares dataframe engines on five sharded queries -- first on
one node, then, as the bigger specialized case, on many:

| tier | sizes | backends |
| --- | --- | --- |
| local | 1-100 GiB | `threads`, `daft`, `ray-data` |
| distributed | 100-1000 GiB | `ray-core`, `dask`, `ray-data`, `daft-ray` |

| backend | where it runs | what it is |
| --- | --- | --- |
| `threads` | in-process | polars-pylance's own `write_lance_fragments` on a thread pool |
| `daft` | in-process | Daft's native multi-threaded runner |
| `ray-core` | processes | polars-pylance shipping pickled `LazyFrame` shards to Ray tasks |
| `dask` | processes | polars-pylance shipping the same shards to Dask workers |
| `ray-data` | processes | Ray Data with lance-ray, running the shared query per batch |
| `daft-ray` | processes | the same Daft plan, executed by Ray |

Ray Data runs in both tiers: on a local Ray cluster below 100 GiB, on the
shared cluster above it. A tier at exactly 100 GiB runs both groups, which
is what makes local-vs-distributed comparable on identical data.

The two in-process backends are the controls: on one node they have no
serialization, no worker to start and no scheduler in the middle, so what a
distributed backend adds over them is what distribution costs. Ray has no
in-process mode to compare against -- `ray.init(local_mode=True)` is "no
longer supported" -- so every Ray row is inter-process by necessity.

No bulk data crosses the scheduler on any of them: workers read and write
Lance storage directly, so this measures compute and IO, not object
transfer. The read case goes further: only row counts and checksums come
back, so it shows distributed read and compute scaling with the output IO
taken out entirely.

## The five cases

| case | kind | what it asks |
| --- | --- | --- |
| `r_agg` | read | filter half the rows away, return count + id checksum per shard |
| `w_filter` | write | selective predicate (`val > 0.9`) plus the wide payload column |
| `w_compute` | write | trig chain per row, separating compute scaling from IO |
| `w_full` | write | full copy of every column, no predicate, maximum IO pressure |
| `w_commit` | write | narrow 1% output at several shard counts; commit time is the variable |

Each case lives in `queries.CASES` as a single Narwhals query, and the same
function runs on every engine: the plan-shipping backends wrap their Polars
shards and unwrap native plans (identical IR, identical pushdown into
Lance), the Daft backends do the same around Daft frames via the
narwhals-daft plugin, and Ray Data maps it over Arrow batches. Only the
Lance SQL string Ray Data pushes into its scanner is spelled separately.
The Daft side needs plugin support for the `w_compute` trig chain
(`sin`/`cos`), which is why the `dataframe` group tracks the plugin's `trig-exprs`
branch until it is released. Verification is a checksum, not a
collect: count and id sum exact, value sum within tolerance, so it stays
affordable at the top of the ladder where collecting whole datasets would
not.

`w_commit` sweeps `DIST_COMMIT_SHARDS` (default `4,16,64`) inside one warmed
process, one record per setting, because the trend of commit time against
fragment count is the measurement. Note the sweep varies *shards*, not file
size: each shard writes at least one fragment whatever `max_rows_per_file`
says, so at small outputs the file-size knob does nothing. `analyse` prints
the trend as fragments → commit share per backend. If a tier has fewer
fragments than a sweep entry asks shards for, the entry degrades to one
shard per fragment and the record says so (`n_shards` is actual, not asked).

Only the query is measured. Every backend builds its cluster and warms its
workers (process start plus the Polars and Lance imports) *before* the timer
starts, because a benchmark of the query should not be a benchmark of Ray's
boot time -- which is several seconds and would otherwise dominate every
small tier and invert the ranking.

```sh
export BENCH_ROOT=/mnt/fast-nvme
uv run bench/polars_lance/gen.py 2000000 4000000 8000000
uv run --group dataframe python -m bench.dataframe.driver.matrix 2000000,4000000,8000000 55
uv run --group dataframe python -m bench.dataframe.analyse "$BENCH_ROOT/dist-results.jsonl"
```

The distributed tier needs a real cluster and storage every worker reaches:

```sh
export BENCH_ROOT=/mnt/fast-nvme          # shared by driver and workers
uv run bench/gen.py 400000000 800000000   # ~200 and ~400 GiB
DIST_CLUSTER=ray://head-node:10001 uv run --group dataframe \
    python -m bench.dataframe.driver.matrix 400000000,800000000 55
```

`run_matrix` takes `<ladder> <cap-GiB>` and writes `dist-results.jsonl`.
Each (backend, case, tier) runs in its own process -- except the commit
sweep, which shares one warmed process per backend -- and every (case,
tier) is verified before the next one starts. A failure is recorded, not
raised, and `analyse` marks it `[!]`.

`BENCH_ROOT` must be real storage: `/tmp` is a RAM disk on many machines,
and Ray also stages under `/tmp`, so point it at `/var/tmp` or a mounted
volume. Ray's uv integration is switched off (`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`
`in `ray_core.py`): left on, every `ray.init` packages the whole driver
worktree for the workers -- slow, flaky, and pointless here, where task
functions ship by value and every dependency is already installed. Override
it in the environment if your workers genuinely need the driver's tree.

Environment variables tune a run: `DIST_CASES` (default all five) selects
cases, `DIST_SHARDS` (default 16) sets the fan-out, `DIST_CHUNK` (default
25000) the streaming batch size, `DIST_COMMIT_SHARDS` (default `4,16,64`)
the commit sweep, `DIST_SLOTS` / `DIST_THREADS` the CPU budget (below), and `BENCH_CAPS=0`
skips the `systemd-run --scope` memory caps where systemd is unavailable.
`DIST_CLUSTER` is the address of an already-running cluster -- required
whenever a tier needs the distributed backends, since silently starting a
local cluster at that scale would benchmark the wrong thing.
Each backend is also runnable on its own against a generated 20k-row
dataset, exercising both its write and its read path, which is the fastest
way to check an environment: `python -m bench.dataframe.backends.dask`.

## Making the comparison fair

Left to themselves the three backends do not take the same machine. Measured
on 8 cores: Ray Core runs 8 tasks in 8 processes and each builds a full
8-thread Polars pool (64 threads for 8 cores), while Dask runs the same 8
tasks in 4 processes sharing 4 pools (32 threads). Both set
`OMP_NUM_THREADS=1`, which Polars ignores -- it sizes its pool from
`POLARS_MAX_THREADS`, at import, so it has to be in the worker's environment
before the worker starts.

`Parallelism` in `backends/__init__.py` therefore states the budget instead of
inheriting it: `DIST_SLOTS` shards run at once with `DIST_THREADS` Polars
threads each, every backend is configured to exactly that, and `slots *
threads` is recorded in each result as `cpu_budget`. `analyse` prints
`UNEQUAL BUDGETS` rather than a ranking if they ever differ. Note that
`DIST_SHARDS` (how the data is cut) and `DIST_SLOTS` (how the machine is)
are independent: more shards than slots is normal and lets the scheduler
even out uneven fragments.

Budget alone is not utilisation, so the measured region also records what
the run consumed. `cpu_seconds`, `cores_busy` and `cpu_utilisation` come
from `/proc/stat`, so they cover every process on the machine, including Ray
workers that are nobody's children.

Memory is measured the same way, for the same reason: `peak_gib` is
`RUSAGE_SELF` and sees only the coordinator, which for an inter-process
backend is close to meaningless. `memory.py` samples one of two whole-machine
sources instead, and says in `mem_source` which it used:

- **cgroup v2**, when the measurement runs inside its own scope. Every
  descendant is charged to that cgroup, so this is exactly the workload and
  nothing else. `run_matrix` gives each measurement a `systemd-run --scope`
  by default, which is what makes this the normal case; verified by the
  cgroup name changing from the login session's to the run's own.
- **the host**, otherwise: `MemTotal - MemAvailable`, which is a fair proxy
  on a dedicated node. `MemAvailable` already discounts reclaimable page
  cache, so reading a 50 GiB dataset does not read as 50 GiB of workload.

So a dedicated EC2 node makes host-wide memory *approximately* the workload,
but only approximately: it still counts every daemon on the box, and it
cannot separate two backends if you run them concurrently. The cgroup path
removes both doubts, which is why the caps are on by default here even
though `BENCH_CAPS=0` is otherwise harmless. Either way `mem_peak_gib` is
the absolute peak and `mem_rise_gib` is the rise above where the run
started; compare backends on the rise. `mem_peak_with_cache_gib` adds the
page cache the cgroup was charged, which is why it is reported separately
rather than mixed in.

`run_matrix` samples the idle machine first and records it as a `baseline`
row, for both CPU and memory.

## Running it at scale

The distributed bench reads the same ladder `../polars_lance/gen.py` writes, from the
same `BENCH_ROOT`, so a large-scale run needs no data of its own -- the
extra `text` and `ts` columns are simply projected away. On the `m8id.4xlarge`
that `../infra/` provisions:

```sh
export BENCH_ROOT=/mnt/nvme
uv run bench/polars_lance/gen.py 48000000 97000000 194000000
uv run --group dataframe python -m bench.dataframe.driver.matrix 48000000,97000000,194000000 55
```

One thing to watch: `gen.py` writes 5M rows per fragment, and fragments are
the ceiling on shards, so a tier only has `rows / 5M` of them to spread. That
is 38 fragments at 194M rows and plenty, but only 9 at 48M, so set
`DIST_SHARDS` no higher than the tier can supply, or regenerate with a
smaller `max_rows_per_file` if you want finer shards at the small end.

One remaining caveat: the local tier compares schedulers, not networks. On
the distributed tier the network is part of what is measured -- which is
why `BENCH_ROOT` and `DIST_CLUSTER` must be reachable from every worker.
Either way the interesting column next to wall time is `plan_bytes` /
`metadata_bytes`, which stay in the kilobytes however big the data gets.

| area | contents |
| --- | --- |
| `queries/` | the five cases as single Narwhals queries (`cases`, `adapters`, `checks`, `harness`) |
| `backends/` | platform code: `threads`, `ray-core`, `dask`, `ray-data`, `daft`, `daft-ray`, plus the registry and the shared plan-shipping pipeline in `sharded` |
| `metrics/` | measurement code: the timed region (`measured`), machine-wide CPU, peak memory |
| `driver/` | `cases` (one measurement per process) and `matrix` (the full ladder) |
| `analyse.py` | renders `dist-results.jsonl` as a comparison table |
