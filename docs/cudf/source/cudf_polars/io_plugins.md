(cudf-polars-io-plugins)=
# Python IO Sources

Polars lets you feed a query from a custom Python function instead of a file, by
registering an *IO plugin* with [`polars.io.plugins.register_io_source`][register-io-source].
cudf-polars executes such sources on the GPU, so you can generate or load data directly into
a query. E.g., to support a custom file format, a remote store, or data produced on the fly.

See the [Polars IO plugins guide][polars-io-plugins] for the full description of
the IO-source contract. This page covers the cudf-polars-specific behavior.

## Plain IO Sources

An IO source is a callable with the signature `(with_columns, predicate, n_rows, batch_size)`.
It yields one or more `polars.DataFrame` batches:

```python
import polars as pl
from polars.io.plugins import register_io_source


def source(with_columns, predicate, n_rows, batch_size):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    if predicate is not None:
        df = df.filter(predicate)
    if n_rows is not None:
        df = df.head(n_rows)
    if with_columns is not None:
        df = df.select(with_columns)

    yield df


lf = register_io_source(source, schema={"a": pl.Int64, "b": pl.Int64})
result = lf.select("a").collect(engine=pl.GPUEngine())
```

The source may yield multiple chunks. cudf-polars concatenates them on the
device. cudf-polars applies pushed Polars predicates on the GPU, so on the GPU
path the source usually receives `predicate=None`. Implementing predicate
handling is still useful when the same source is collected with the default
Polars CPU engine.

## Schema Validation

The columns a source emits (after applying `with_columns`) must match the
registered schema in name, order, and dtype. cudf-polars validates the produced
output against that schema and raises [`polars.exceptions.SchemaError`](https://docs.pola.rs/api/python/stable/reference/exceptions.html)
on a mismatch.

Polars itself only validates when `register_io_source(..., validate_schema=True)`
is used, but that flag is not carried into the GPU plan, so cudf-polars validates
unconditionally. A source that deliberately yields a dtype different from its
declared schema (only valid with `validate_schema=False`) therefore cannot run on
the GPU and must be collected with the default Polars CPU engine. The check is
metadata-only (column names and dtypes) and does not touch the data.

## Rank-Aware Sources

With a multi-GPU streaming engine (see {doc}`engines`), every rank runs the scan
function. A plain Python source has no rank information, so if it ran on every
rank it would usually emit duplicate rows. cudf-polars avoids that by executing
plain sources on rank 0 only.

For distributed loading, use a rank-aware source. Subclass
{class}`~cudf_polars.streaming.rank_aware_source.RankAwareSource`; its
`__call__` method follows the `register_io_source` contract and adds two
optional trailing arguments:

- `rank`: the zero-based rank running the source.
- `nranks`: the total number of ranks in the query.

Both are `None` for single-rank execution, the in-memory cudf-polars engine, and
the default Polars CPU engine. In multi-rank execution, each rank must emit only
its local rows so the union across ranks reconstructs the dataset exactly once.

```python
import polars as pl
from polars.io.plugins import register_io_source
from cudf_polars.streaming.rank_aware_source import RankAwareSource


class PartitionedFrame(RankAwareSource):
    def __init__(self, frame):
        self.frame = frame

    def __call__(
        self,
        with_columns,
        predicate,
        n_rows,
        batch_size,
        rank=None,
        nranks=None,
    ):
        if rank is None or nranks is None:
            df = self.frame
        else:
            df = self.frame.gather_every(nranks, offset=rank)

        if predicate is not None:
            df = df.filter(predicate)
        if n_rows is not None:
            df = df.head(n_rows)
        if with_columns is not None:
            df = df.select(with_columns)

        yield df


source = PartitionedFrame(pl.DataFrame({"a": range(10)}))
lf = register_io_source(source, schema={"a": pl.Int64})
```

A batch may be a host `polars.DataFrame` or an already-GPU-resident
`cudf_polars.containers.DataFrame`. Returning the latter skips the
host-to-device copy, but that source can only be collected with a cudf-polars
engine because the default Polars engine cannot consume GPU frames.

### Loading One Partition Per Rank

For the common case where each rank has one loader,
{meth}`~cudf_polars.streaming.rank_aware_source.RankAwareSource.from_loaders`
registers the source for you. Pass a `{rank: loader}` mapping and a schema. Each
loader is a zero-argument callable that returns either a `polars.DataFrame` or a
`cudf_polars.containers.DataFrame`:

```python
from functools import partial

import polars as pl
from cudf_polars.streaming.rank_aware_source import RankAwareSource


def load_partition(partition_id):
    return pl.read_parquet(f"s3://bucket/part-{partition_id}.parquet")


lf = RankAwareSource.from_loaders(
    {
        0: partial(load_partition, 0),
        1: partial(load_partition, 1),
    },
    schema={"a": pl.Int64},
)
```

The helper handles rank routing, ranks without a loader, and single-rank
concatenation. Loaders are serialized with the query plan and invoked lazily
inside the cudf-polars executor thread on the owning rank. Prefer module-level
functions or `functools.partial` objects over lambdas when the source must run
on remote Ray or Dask workers.

## Row-Limit Pushdown

Polars can push `head` / `limit` operations into a Python scan as `n_rows`.
cudf-polars does not currently support this pushdown and rejects such plans
during translation. Standard GPU fallback behavior applies: by default the query
falls back to the Polars CPU engine, and in raise-on-fail mode cudf-polars raises
`NotImplementedError`.

## API

```{eval-rst}
.. autoclass:: cudf_polars.streaming.rank_aware_source.RankAwareSource
   :members: from_loaders
   :special-members: __call__
```

[register-io-source]: https://docs.pola.rs/api/python/stable/reference/api/polars.io.plugins.register_io_source.html
[polars-io-plugins]: https://docs.pola.rs/user-guide/plugins/io_plugins/
