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
from typing import cast

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


@pytest.mark.parametrize(
    ("arrow_schema", "custom_staging"),
    [(False, False), (True, False), (False, True)],
    ids=["defaults", "arrow schema", "custom staging"],
)
def test_round_trip(
    tmp_path: Path,
    lance_uri: str,
    arrow_schema: bool,  # noqa: FBT001 - a pytest parameter
    custom_staging: bool,  # noqa: FBT001 - a pytest parameter
) -> None:
    """Workers write fragments and publish nothing; one commit publishes them all.

    The commit also removes the staging prefix.
    """
    out = str(tmp_path / "out.lance")
    lf = _transformed(lance_uri)
    schema = lf.collect_schema().to_arrow() if arrow_schema else lf
    staging_uri = str(tmp_path / "staging-elsewhere") if custom_staging else None

    staged = stage_lance_sink(out, schema, staging_uri=staging_uri)
    _run(staged, lf, chunk_size=2_000)

    with pytest.raises(ValueError, match="was not found"):
        lance.dataset(out)
    assert len(staged.staged_fragments()) > 1
    if staging_uri is not None:
        assert staged.staging_uri.startswith(staging_uri)
    staging_dir = Path(staged.staging_uri)
    assert staging_dir.exists()

    dataset = staged.commit()
    assert not staging_dir.exists()
    assert len(dataset.get_fragments()) > 1
    assert_frame_equal(
        scan_lance(out).collect(engine="streaming").sort("id"),
        lf.collect(engine="streaming").sort("id"),
    )


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
    assert a.commit(cleanup=True).count_rows() == lf.collect(engine="streaming").height
    assert b.staged_fragments(), "b's staging survived a's cleanup"


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


# -- what has to survive the trip to a worker -------------------------------


def test_callback_pickles_by_reference(tmp_path: Path, lance_uri: str) -> None:
    """The plan carries data, not code: workers import polars-pylance themselves.

    That is why :func:`polars_pylance.cloud.requirements_txt` lists the package.
    """
    out = str(tmp_path / "pickled.lance")
    lf = _transformed(lance_uri)
    staged = stage_lance_sink(out, lf, max_rows_per_file=1_000)
    blob = pickle.dumps(staged.callback)
    assert len(blob) < 4_000

    revived = pickle.loads(blob)
    assert revived == staged.callback
    lf.sink_batches(revived, chunk_size=5_000, engine="streaming", lazy=False)
    assert staged.commit().count_rows() == lf.collect(engine="streaming").height


def test_callback_survives_into_a_cloud_plan(tmp_path: Path, lance_uri: str) -> None:
    """The whole premise: the writer ships inside the plan, so it runs on the workers.

    polars needs `cloudpickle`, a test dependency, to serialize the callback.
    """
    from polars._utils.cloud import prepare_cloud_plan

    lf = _transformed(lance_uri)
    staged = stage_lance_sink(str(tmp_path / "planned.lance"), lf)
    # Annotated as `bytes`, but returns the plan with its optimization flags.
    plan, _ = cast(
        "tuple[bytes, object]",
        prepare_cloud_plan(lf.sink_batches(staged.callback, lazy=True)),
    )

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
