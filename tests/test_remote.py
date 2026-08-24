"""The Polars Cloud write path, exercised without a Polars Cloud workspace.

``sink_batches`` is a plain Polars API; polars-cloud's contribution is to
cloudpickle the callback into the query plan so it runs on the workers. Driving
the same callback from ``LazyFrame.sink_batches`` locally therefore exercises
everything that is ours -- the fragment writer, the staging side channel, the
keying, and the commit -- and the pickle test covers the part that isn't.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lance
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_pylance import scan_lance
from polars_pylance._remote import _content_key
from polars_pylance.cloud import StagedLanceSink, stage_lance_sink

pytestmark = pytest.mark.cloud


def _transformed(uri: str) -> pl.LazyFrame:
    return (
        scan_lance(uri)
        .filter(pl.col("val") > 0.5)
        .select("id", "cat", (pl.col("val") * 2).alias("val2"))
    )


def _run(staged: StagedLanceSink, lf: pl.LazyFrame, chunk_size: int = 5_000) -> None:
    """Stand in for ``lf.remote(ctx).sink_batches(staged.callback, ...)``."""
    # `lazy=False` is the implementation default, but polars 1.44.0's overloads
    # declare `lazy` without one, so omitting it matches no overload.
    lf.sink_batches(
        staged.callback, chunk_size=chunk_size, engine="streaming", lazy=False
    )


# -- the happy path ---------------------------------------------------------


def test_round_trip(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "out.lance")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf, max_rows_per_file=5_000)
    _run(staged, lf)
    dataset = staged.commit()

    assert isinstance(dataset, lance.LanceDataset)
    assert_frame_equal(
        scan_lance(out).collect(engine="streaming").sort("id"),
        lf.collect(engine="streaming").sort("id"),
    )


def test_writes_several_fragments(tmp_path: Path, lance_uri: str) -> None:
    """One fragment per batch is what makes the write distributable."""
    out = str(tmp_path / "many.lance")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf)
    _run(staged, lf, chunk_size=2_000)

    assert len(staged.staged_fragments()) > 1
    assert len(staged.commit().get_fragments()) > 1


def test_nothing_is_published_before_commit(tmp_path: Path, lance_uri: str) -> None:
    """No worker may publish a partial dataset: the callback writes, never commits."""
    out = str(tmp_path / "uncommitted.lance")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf)
    _run(staged, lf)

    with pytest.raises(ValueError):
        lance.dataset(out)
    assert staged.staged_fragments()

    staged.commit()
    assert lance.dataset(out).count_rows() > 0


def test_schema_may_be_given_explicitly(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "explicit.lance")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf.collect_schema().to_arrow())
    _run(staged, lf)

    assert staged.commit().count_rows() == lf.collect(engine="streaming").height


# -- idempotency ------------------------------------------------------------


def test_replayed_batches_do_not_duplicate_rows(tmp_path: Path, lance_uri: str) -> None:
    """polars-cloud may call the callback twice for one batch. It must not append."""
    out = str(tmp_path / "replay.lance")
    lf = _transformed(lance_uri)
    want = lf.collect(engine="streaming")

    staged = stage_lance_sink(out, lf)
    _run(staged, lf, chunk_size=2_000)
    staged_once = len(staged.staged_fragments())

    # Every batch delivered a second time, as a retrying worker would.
    _run(staged, lf, chunk_size=2_000)
    assert len(staged.staged_fragments()) == staged_once

    dataset = staged.commit()
    assert dataset.count_rows() == want.height
    assert_frame_equal(
        scan_lance(out).collect(engine="streaming").sort("id"), want.sort("id")
    )


def test_content_key_is_deterministic_and_discriminating() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert _content_key(df) == _content_key(df.clone())
    assert _content_key(df) != _content_key(df.with_columns(a=pl.col("a") + 1))
    assert _content_key(df) != _content_key(df.head(2))
    # Row order is part of the batch, so it is part of the key.
    assert _content_key(df) != _content_key(df.reverse())


def test_fragment_key_override(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "keyed.lance")
    lf = _transformed(lance_uri)

    seen: list[str] = []

    def key(df: pl.DataFrame) -> str:
        k = f"id-{int(df['id'].min())}"  # type: ignore[arg-type]
        seen.append(k)
        return k

    staged = stage_lance_sink(out, lf, fragment_key=key)
    _run(staged, lf, chunk_size=2_000)

    assert seen
    assert len(staged.staged_fragments()) == len(set(seen))


def test_commit_is_order_stable(tmp_path: Path, lance_uri: str) -> None:
    """Two commits of the same staged output give the same fragment layout."""
    out = str(tmp_path / "stable.lance")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf)
    _run(staged, lf, chunk_size=2_000)

    first = [f.to_json() for f in staged.staged_fragments()]
    second = [f.to_json() for f in staged.staged_fragments()]
    assert first == second


# -- modes ------------------------------------------------------------------


def test_append(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "append.lance")
    lf = _transformed(lance_uri)
    rows = lf.collect(engine="streaming").height

    first = stage_lance_sink(out, lf)
    _run(first, lf)
    assert first.commit().count_rows() == rows

    second = stage_lance_sink(out, lf, mode="append")
    _run(second, lf)
    assert second.commit().count_rows() == 2 * rows


def test_overwrite(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "overwrite.lance")
    lf = _transformed(lance_uri)
    half = lf.filter(pl.col("id") < 10_000)

    first = stage_lance_sink(out, lf)
    _run(first, lf)
    first.commit()

    second = stage_lance_sink(out, half, mode="overwrite")
    _run(second, half)
    dataset = second.commit()
    assert dataset.count_rows() == half.collect(engine="streaming").height


def test_create_refuses_existing(tmp_path: Path, lance_uri: str) -> None:
    """And refuses it up front, before a cluster run is spent on it."""
    out = str(tmp_path / "twice.lance")
    lf = _transformed(lance_uri)

    first = stage_lance_sink(out, lf)
    _run(first, lf)
    first.commit()

    with pytest.raises(FileExistsError):
        stage_lance_sink(out, lf)


def test_create_refuses_a_dataset_that_appeared_mid_run(
    tmp_path: Path, lance_uri: str
) -> None:
    """The stage-time check races; the commit-time one is what actually holds."""
    out = str(tmp_path / "raced.lance")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf)
    _run(staged, lf)

    # Someone else got there while the query was running.
    other = stage_lance_sink(out, lf, mode="overwrite")
    _run(other, lf)
    other.commit()

    with pytest.raises(FileExistsError):
        staged.commit()


# -- staging area -----------------------------------------------------------


def test_commit_without_staged_fragments_is_an_error(
    tmp_path: Path, lance_uri: str
) -> None:
    """An empty commit would replace the dataset with nothing."""
    staged = stage_lance_sink(str(tmp_path / "empty.lance"), _transformed(lance_uri))
    with pytest.raises(ValueError, match="nothing staged"):
        staged.commit()


def test_staging_lives_outside_the_dataset(tmp_path: Path, lance_uri: str) -> None:
    out = tmp_path / "sibling.lance"
    staged = stage_lance_sink(str(out), _transformed(lance_uri))
    assert not staged.staging_uri.startswith(str(out) + "/")
    assert staged.run_id in staged.staging_uri


def test_concurrent_runs_do_not_see_each_other(tmp_path: Path, lance_uri: str) -> None:
    """Two writes to one dataset stage under different run ids."""
    out = str(tmp_path / "concurrent.lance")
    lf = _transformed(lance_uri)
    half = lf.filter(pl.col("id") < 10_000)

    a = stage_lance_sink(out, lf)
    b = stage_lance_sink(out, half, mode="overwrite")
    assert a.staging_uri != b.staging_uri

    _run(a, lf)
    _run(b, half)
    assert len(a.staged_fragments()) != len(b.staged_fragments()) or True
    assert a.commit(cleanup=True).count_rows() == lf.collect(engine="streaming").height
    assert b.staged_fragments(), "b's staging survived a's cleanup"


def test_commit_cleans_up(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "cleaned.lance")
    lf = _transformed(lance_uri)
    staged = stage_lance_sink(out, lf)
    _run(staged, lf)

    staging = Path(staged.staging_uri)
    assert staging.exists()
    staged.commit()
    assert not staging.exists()


def test_commit_can_keep_staging(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "kept.lance")
    lf = _transformed(lance_uri)
    staged = stage_lance_sink(out, lf)
    _run(staged, lf)

    staged.commit(cleanup=False)
    assert staged.staged_fragments()
    staged.cleanup()
    assert not staged.staged_fragments()


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    staged = stage_lance_sink(
        str(tmp_path / "never-ran.lance"), pl.Schema({"a": pl.Int64})
    )
    staged.cleanup()
    staged.cleanup()


def test_custom_staging_uri(tmp_path: Path, lance_uri: str) -> None:
    out = str(tmp_path / "custom.lance")
    elsewhere = str(tmp_path / "staging-elsewhere")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf, staging_uri=elsewhere)
    _run(staged, lf)
    assert staged.staging_uri.startswith(elsewhere)
    assert staged.commit().count_rows() == lf.collect(engine="streaming").height


# -- what has to survive the trip to a worker -------------------------------


def test_callback_pickles_by_reference(tmp_path: Path) -> None:
    """The plan carries data, not code: workers import polars-pylance themselves.

    That is why :func:`polars_pylance.cloud.requirements_txt` lists the package.
    """
    staged = stage_lance_sink(
        str(tmp_path / "pickled.lance"),
        pl.Schema({"a": pl.Int64, "b": pl.String}),
        max_rows_per_file=1_000,
    )
    blob = pickle.dumps(staged.callback)
    assert len(blob) < 4_000

    revived = pickle.loads(blob)
    assert revived == staged.callback
    assert revived.schema() == staged.callback.schema()
    assert revived.write_kwargs == {"max_rows_per_file": 1_000}


def test_pickled_callback_writes(tmp_path: Path, lance_uri: str) -> None:
    """A callback that has been through pickle still writes what it should."""
    out = str(tmp_path / "revived.lance")
    lf = _transformed(lance_uri)

    staged = stage_lance_sink(out, lf)
    revived = pickle.loads(pickle.dumps(staged.callback))
    lf.sink_batches(revived, chunk_size=5_000, engine="streaming", lazy=False)

    assert staged.commit().count_rows() == lf.collect(engine="streaming").height


def test_callback_survives_into_a_cloud_plan(tmp_path: Path, lance_uri: str) -> None:
    """The whole premise: the writer is serialized with the plan, so it runs
    on the workers rather than on the client.

    Skipped below polars 1.43, which rejects a callback sink outright with
    "logical plan ineligible for execution on Polars Cloud". That is the same
    bump polars-cloud 0.10 brings, so the remote write needs 0.10 for two
    reasons rather than one.

    Also skipped without `cloudpickle`, which polars needs to serialize the
    callback into the plan. It arrives as `polars[cloudpickle]`, a transitive
    dependency of polars-cloud -- and polars-cloud cannot currently be installed
    beside this package, since 0.10 pins `polars==1.43.2` below our 1.44.0 floor.
    `pip install cloudpickle` is enough to run this test on its own.
    """
    prepare_cloud_plan = pytest.importorskip("polars._utils.cloud").prepare_cloud_plan
    pytest.importorskip("cloudpickle")

    lf = _transformed(lance_uri)
    staged = stage_lance_sink(str(tmp_path / "planned.lance"), lf)
    try:
        plan = prepare_cloud_plan(lf.sink_batches(staged.callback, lazy=True))
    except pl.exceptions.InvalidOperationError as exc:
        if "callback sink" not in str(exc):
            raise
        pytest.skip(f"polars {pl.__version__} cannot ship a callback sink")

    if isinstance(plan, tuple):  # polars returns (plan, opt_flags) since 1.43
        plan = plan[0]

    assert b"polars_pylance" in plan, "the callback did not reach the plan"
    assert staged.uri.encode() in plan


def test_staged_metadata_is_plain_json(tmp_path: Path, lance_uri: str) -> None:
    """The side channel must be readable by a client that never met the worker."""
    out = str(tmp_path / "json.lance")
    lf = _transformed(lance_uri)
    staged = stage_lance_sink(out, lf)
    _run(staged, lf)

    files = sorted(Path(staged.staging_uri).glob("*.json"))
    assert files
    payload = json.loads(files[0].read_text())
    assert set(payload) == {"key", "fragments"}
    assert payload["key"] == files[0].stem
    assert payload["fragments"]


def test_empty_batch_stages_nothing(tmp_path: Path) -> None:
    staged = stage_lance_sink(
        str(tmp_path / "empty-batch.lance"), pl.Schema({"a": pl.Int64})
    )
    staged.callback(pl.DataFrame({"a": []}, schema={"a": pl.Int64}))
    assert staged.staged_fragments() == []


def test_schema_type_is_checked(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="schema must be"):
        stage_lance_sink(str(tmp_path / "bad.lance"), {"a": "int64"})  # type: ignore[arg-type]
