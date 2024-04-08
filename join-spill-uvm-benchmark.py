# Extended version of <https://github.com/rapidsai/db-benchmark/issues/9>

import gc
import timeit
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, Union

import cupy as cp
import numpy as np
import rmm
import typer
from rmm.mr import (CudaAsyncMemoryResource, CudaMemoryResource,
                    ManagedMemoryResource, PoolMemoryResource)
from typing_extensions import Annotated

import cudf
from cudf.core.buffer.spill_manager import get_global_manager

class Base(str, Enum):
    CUDA = "cuda"
    CUDA_ASYNC = "async"
    MANAGED = "managed"


MemoryResource: TypeAlias = Union[
    CudaAsyncMemoryResource,
    CudaMemoryResource,
    ManagedMemoryResource,
    PoolMemoryResource,
]

VALID_STRATEGIES: dict[tuple[Base, bool, bool], tuple[MemoryResource, bool, bool]] = {
    (Base.CUDA, True, True): (CudaMemoryResource, True, True),
    (Base.CUDA, False, True): (CudaMemoryResource, False, True),
    (Base.CUDA, True, False): (CudaMemoryResource, True, False),
    (Base.CUDA, False, False): (CudaMemoryResource, False, False),
    (Base.CUDA_ASYNC, True, False): (CudaAsyncMemoryResource, True, False),
    (Base.CUDA_ASYNC, False, False): (CudaAsyncMemoryResource, False, False),
    (Base.MANAGED, False, True): (ManagedMemoryResource, False, True),
    (Base.MANAGED, False, False): (ManagedMemoryResource, False, False),
    (Base.MANAGED, True, True): (ManagedMemoryResource, True, True),
}

def expand_callback(size: int) -> bool:
    manager = get_global_manager()
    # print("expand_callback() - manager: ", manager)
    if manager is None:
        return False
    ret = manager.spill_device_memory(nbytes=size)
    # print(f"expand_callback() - size {size}, cudf spilled: {ret}")
    return ret


def allocation_strategy(
    base: Base, spill: bool, pool: bool
) -> tuple[MemoryResource, bool]:

    try:
        mr, spill, pool = VALID_STRATEGIES[(base, spill, pool)]
        if pool:
            mr = PoolMemoryResource(mr(), initial_pool_size=30*(2**30), maximum_pool_size=None)
            if base == Base.MANAGED and spill and hasattr(mr, "set_expand_callback"):
                mr.set_expand_callback(expand_callback)
            return mr, spill
        else:
            return mr(), spill
    except KeyError:
        raise RuntimeError(f"Allocation strategy {base}-{spill=}-{pool=} is not valid")


# Set to false if random number generation blows through memory limit
USE_CUPY = True
# USE_CUPY = False
if USE_CUPY:
    rng = cp.random._generator.RandomState(seed=108)
    xp = cp
else:
    rng = np.random.default_rng(seed=108)
    xp = np


@dataclass
class Key:
    values: xp.ndarray
    x: slice
    l: slice
    r: slice

    def __init__(self, n: int):
        from cudf.utils.dtypes import min_unsigned_type
        Nleft = n // 10

        Ncommon = n - Nleft
        dtype = min_unsigned_type(n + Nleft - 1)
        key = rng.permutation(xp.arange(n + Nleft, dtype=dtype))
        self.values = key
        self.x = slice(None, Ncommon)
        self.l = slice(Ncommon, Ncommon + Nleft)
        self.r = slice(Ncommon + Nleft, None)

    @property
    def lhs(self):
        return xp.concatenate([self.values[self.x], self.values[self.l]])

    @property
    def rhs(self):
        return xp.concatenate([self.values[self.x], self.values[self.r]])

    def permute_sample(self, val: xp.ndarray, n: int):
        if len(val) > n:
            raise RuntimeError
        if len(val) == n:
            return rng.permutation(val)
        return rng.permutation(
            xp.concatenate([val, rng.choice(val, size=(n - len(val)), replace=True)])
        )


def create_tables(P: int, N: int, *, string_categoricals: bool):
    key1 = Key(N // 10**6)
    id1 = cudf.Series(key1.permute_sample(key1.lhs, N))
    rid1 = cudf.Series(key1.permute_sample(key1.rhs, P))
    del key1
    key2 = Key(N // 10**3)
    id2 = cudf.Series(key2.permute_sample(key2.lhs, N))
    rid2 = cudf.Series(key2.permute_sample(key2.rhs, P))
    del key2
    key3 = Key(N)
    id3 = cudf.Series(key3.permute_sample(key3.lhs, N))
    del key3

    v1 = cudf.Series(rng.uniform(low=0, high=100, size=(N,)).astype("float32"))
    v2 = cudf.Series(rng.uniform(low=0, high=100, size=(P,)).astype("float32"))

    left = cudf.DataFrame({"id1": id1, "id2": id2, "id3": id3, "v1": v1})
    right = cudf.DataFrame({"id1": rid1, "id2": rid2, "v2": v2})

    left["id4"] = left["id1"].astype("category")
    right["id4"] = right["id1"].astype("category")
    # Can't do this, because decatting the columns for merge overflows
    # string column limits
    if string_categoricals:
        left["id4"] = left["id4"].cat.set_categories(
            [f"id{i}" for i in left["id4"].cat.categories.values_host],
            rename=True,
            ordered=False,
        )
        right["id4"] = right["id4"].cat.set_categories(
            [f"id{i}" for i in right["id4"].cat.categories.values_host],
            rename=True,
            ordered=False,
        )
    left["id5"] = left["id2"].astype("category")
    right["id5"] = right["id2"].astype("category")
    if string_categoricals:
        left["id5"] = left["id5"].cat.set_categories(
            [f"id{i}" for i in left["id5"].cat.categories.values_host],
            rename=True,
            ordered=False,
        )
        right["id5"] = right["id5"].cat.set_categories(
            [f"id{i}" for i in right["id5"].cat.categories.values_host],
            rename=True,
            ordered=False,
        )

    # TODO: This should be included in the left frame, but ignoring it
    # for now since it is so nonsense
    # This is a nonsense categorical!
    # left["id6"] = left["id3"].astype("category", ordered=False)
    # if string_categoricals:
    #     left["id6"].cat.set_categories(
    #         [f"id{i}" for i in left["id6"].cat.categories.values_host],
    #         rename=True,
    #         ordered=False,
    #     )
    return left, right

import traceback
def with_timing(fn, *, query) -> int:
    total = 0
    for i in range(5):
        gc.collect()
        try:
            start = timeit.default_timer()
            val = fn()
            end = timeit.default_timer()
            total += end - start
            del val
            # if i == 1:
            print(f"{query} repeat={i+1}: {end - start:.2f}s")
        except Exception as e:
            print(f"Query {query} failed: \n{traceback.format_exc()}")
        finally:
            gc.collect()
    return total


def run_queries(left: cudf.DataFrame, right: cudf.DataFrame) -> int:
    total = 0
    total += with_timing(
        lambda: left.merge(right, on="id2", how="inner"), query="medium inner on int"
    )
    total += with_timing(
        lambda: left.merge(right, on="id2", how="left"), query="medium outer on int"
    )
    total += with_timing(
        lambda: left.merge(right, on="id5", how="inner"), query="medium inner on factor"
    )
    return total


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    n: Annotated[int, typer.Argument(help="Number of rows in left table")],
    base_memory_resource: Annotated[
        Base, typer.Option(help="Base memory resource")
    ] = Base.CUDA,
    use_pool: Annotated[bool, typer.Option(help="Use RMM memory pool?")] = False,
    use_spilling: Annotated[bool, typer.Option(help="Use cudf spilling?")] = False,
    string_categoricals: Annotated[
        bool, typer.Option(help="Should categories use string names?")
    ] = False,
):
    mr, spill = allocation_strategy(base_memory_resource, use_spilling, use_pool)
    cudf.set_option("spill", spill)
    cudf.set_option("spill_on_demand", True)
    rmm.mr.set_current_device_resource(mr)
    print("Running experiment")
    print("------------------")
    print(f"{mr=}")
    print(f"{use_pool=}")
    print(f"{use_spilling=}")
    print(f"{string_categoricals=}")
    try:
        left, right = create_tables(
            n // 10**3, n, string_categoricals=string_categoricals
        )
    except Exception as e:
        print(f"Failed generating tables: {e}")
        return

    nbytes_left = left.memory_usage().sum() / 1024 ** 3
    nbytes_right = right.memory_usage().sum() / 1024 ** 3
    print(f"Left table has {len(left)} rows and is {nbytes_left:.2f} GiB")
    print(f"Right table is {len(right)} rows and is {nbytes_right:.2f} GiB")
    total = run_queries(left, right)
    print(f"Total time: {total:0.2f}")

    if get_global_manager() is not None:
        print(get_global_manager().statistics)


if __name__ == "__main__":
    app()
