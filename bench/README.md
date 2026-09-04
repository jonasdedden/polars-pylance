# Benchmarks

Two areas, one shared rig:

- [`polars_lance`](polars_lance/README.md) compares polars-pylance against
  `polars-lance` across a nine-tier size ladder, measuring wall time and peak
  RSS for every case.
- [`dataframe`](dataframe/README.md) compares dataframe engines on five
  sharded queries: polars-pylance against Daft and Ray Data on one node, and
  -- as the bigger specialized case -- polars-pylance on Ray Core and Dask,
  Daft on Ray, and Ray Data on many.
- [`infra`](infra/run_remote.sh) provisions the single EC2 instance with local
  NVMe both areas run on. Multi-instance setups will come later; for now the
  multi-node backends are exercised through their single-node smoke tests.
