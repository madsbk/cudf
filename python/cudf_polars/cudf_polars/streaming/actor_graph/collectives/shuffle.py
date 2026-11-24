# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc
import pylibcudf.partitioning
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from cudf_streaming.partition_utils import (
    partition_and_pack_cost as py_partition_and_pack_cost,
    split_and_pack as py_split_and_pack,
    split_and_pack_cost as py_split_and_pack_cost,
    unpack_and_concat as py_unpack_and_concat,
    unpack_and_concat_cost as py_unpack_and_concat_cost,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.nodes import shutdown_on_error
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    _is_already_partitioned,
    recv_metadata,
    send_metadata,
)
from cudf_polars.streaming.shuffle import Shuffle
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.channel import Channel
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.core import SubNetGenerator
    from cudf_polars.typing import Schema


class ShuffleManager:
    """
    ShufflerAsync manager.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    num_partitions
        The number of partitions to shuffle into.
    collective_id
        The collective ID.
    partition_assignment, optional
        How to assign partition IDs to ranks: ROUND_ROBIN (default) or
        CONTIGUOUS. Use CONTIGUOUS for sort so each rank gets adjacent
        partition IDs and concatenation order matches global order.
    """

    class Inserter:
        """
        Context manager for the insert phase of a shuffle operation.

        Obtained via :meth:`ShuffleManager.inserting`. On exit, signals
        the end of insertion to all ranks by calling ``insert_finished()``.

        Parameters
        ----------
        manager
            The shuffle manager to insert into.
        """

        def __init__(self, manager: ShuffleManager):
            self._manager = manager

        async def insert_hash(
            self, chunk: TableChunk, columns_to_hash: tuple[int, ...]
        ) -> None:
            """Partition chunk by hash and insert into the shuffler."""
            columns = chunk.table_view().columns()
            await self._insert_hashed(
                chunk,
                lambda: plc.Table([columns[i] for i in columns_to_hash]),
                keys_allocate=False,
            )

        async def insert_hash_keys(
            self, chunk: TableChunk, keys: tuple[NamedExpr, ...], schema: Schema
        ) -> None:
            """Partition chunk by hash of ``keys`` evaluated over it, and insert."""
            await self._insert_hashed(
                chunk,
                lambda: _evaluate_key_table(chunk, keys, schema),
                keys_allocate=True,
            )

        async def _insert_hashed(
            self,
            chunk: TableChunk,
            make_keys: Callable[[], plc.Table],
            *,
            keys_allocate: bool,
        ) -> None:
            """
            Partition ``chunk`` by the hash of the keys ``make_keys`` returns, and insert.

            If the device memory available for reservation covers reordering
            the whole chunk at once, the chunk is reordered with
            ``hash_partition`` and packed. Otherwise only a gather map is
            computed, and the chunk is gathered, packed and inserted in batches
            of partitions sized to the available memory.

            Parameters
            ----------
            chunk
                The chunk to partition.
            make_keys
                Returns the table of keys to hash, row-aligned with ``chunk``.
            keys_allocate
                Whether ``make_keys`` allocates a new table, charged as one
                chunk. This is an approximation, not a bound. A key expression
                that expands its input evaluates to more than the chunk's
                packed size and under-reserves.
            """
            context = self._manager.context
            br = context.br()
            table = chunk.table_view()
            stream = chunk.stream
            chunk_nbytes = py_split_and_pack_cost(table, stream, br)
            keys_nbytes = chunk_nbytes if keys_allocate else 0
            # The keys, the reorder and the pack.
            whole_nbytes = keys_nbytes + 2 * chunk_nbytes
            if _device_memory_available(br) >= whole_nbytes:
                reservation = await reserve_memory(
                    context,
                    whole_nbytes,
                    # The chunk's data moves into shuffler-owned packed buffers,
                    # nothing lasting is added.
                    net_memory_delta=0,
                )
                with opaque_memory_usage(reservation.split(keys_nbytes)):
                    keys = make_keys()
                with opaque_memory_usage(reservation.split(chunk_nbytes)):
                    reordered, offsets = plc.partitioning.hash_partition(
                        table,
                        keys,
                        self._manager.num_partitions,
                        stream=stream,
                        mr=br.device_mr,
                    )
                del keys
                self._manager.shuffler.insert(
                    py_split_and_pack(
                        table=reordered,
                        splits=list(offsets[1:-1]),
                        stream=stream,
                        br=br,
                        reservation=reservation,
                    )
                )
                return

            reservation = await reserve_memory(
                context,
                keys_nbytes + _hash_partition_indices_cost(table.num_rows()),
                net_memory_delta=0,
            )
            with opaque_memory_usage(reservation.split(keys_nbytes)):
                keys = make_keys()
            with opaque_memory_usage(reservation):
                gather_map, offsets = self._hash_partition_indices(keys, stream)
            # Free the keys before any of the chunk is copied.
            del keys
            await self._gather_and_insert(
                table, gather_map, offsets, chunk_nbytes, stream
            )

        def _hash_partition_indices(
            self, keys: plc.Table, stream: Stream
        ) -> tuple[plc.Column, list[int]]:
            """Hash ``keys`` into a gather map grouped by partition and its offsets."""
            return plc.partitioning.hash_partition_indices(
                keys,
                self._manager.num_partitions,
                # `cudf::hash_partition` doesn't keep the input order either.
                plc.partitioning.PartitionOrder.UNSTABLE,
                stream=stream,
                mr=self._manager.context.br().device_mr,
            )

        async def _gather_and_insert(
            self,
            table: plc.Table,
            gather_map: plc.Column,
            offsets: list[int],
            table_nbytes: int,
            stream: Stream,
        ) -> None:
            """
            Gather, pack and insert ``table`` in batches of consecutive partitions.

            Each batch is sized, by an estimate from its row count, so that its
            gather and its pack fit in half each of the device memory available
            for reservation when the batch starts, but at least
            ``_MIN_PACK_BATCH_NBYTES``. A partition larger than that is a batch
            of its own. A batch is inserted into the shuffler before the next
            one is gathered, and each batch reserves its own memory.

            Parameters
            ----------
            table
                The table to partition.
            gather_map
                Row indices of ``table`` grouped by partition.
            offsets
                ``num_partitions + 1`` offsets into ``gather_map``, where partition
                ``i`` is ``gather_map[offsets[i]:offsets[i + 1]]``.
            table_nbytes
                The packed size of ``table``.
            stream
                CUDA stream that ``table`` and ``gather_map`` are ordered on.
            """
            context = self._manager.context
            br = context.br()
            num_rows = table.num_rows()
            num_partitions = len(offsets) - 1
            begin = 0
            while begin < num_partitions:
                max_batch_nbytes = max(
                    _device_memory_available(br) // 2, _MIN_PACK_BATCH_NBYTES
                )
                end = _batch_end(
                    offsets,
                    begin,
                    max(1, num_rows * max_batch_nbytes // max(table_nbytes, 1)),
                )
                (batch_map,) = plc.copying.slice(
                    gather_map, [offsets[begin], offsets[end]], stream=stream
                )
                # Estimated from the row count, since rows differ in size.
                gather_nbytes = -(
                    -table_nbytes * (offsets[end] - offsets[begin]) // max(num_rows, 1)
                )
                with opaque_memory_usage(
                    await reserve_memory(context, gather_nbytes, net_memory_delta=0)
                ):
                    batch = plc.copying.gather(
                        table,
                        batch_map,
                        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                        stream=stream,
                        mr=br.device_mr,
                    )
                reservation = await reserve_memory(
                    context,
                    py_split_and_pack_cost(batch, stream, br),
                    # The batch's data moves into shuffler-owned packed buffers,
                    # nothing lasting is added.
                    net_memory_delta=0,
                )
                packed = py_split_and_pack(
                    table=batch,
                    splits=[offsets[i] - offsets[begin] for i in range(begin + 1, end)],
                    stream=stream,
                    br=br,
                    reservation=reservation,
                )
                del batch
                self._manager.shuffler.insert(
                    {begin + pid: data for pid, data in packed.items()}
                )
                begin = end

        async def insert_split(self, chunk: TableChunk, splits: list[int]) -> None:
            """Split chunk at the given indices and insert into the shuffler."""
            br = self._manager.context.br()
            reservation = await reserve_memory(
                self._manager.context,
                py_split_and_pack_cost(chunk.table_view(), chunk.stream, br),
                # The chunk's data moves into shuffler-owned packed buffers,
                # nothing lasting is added.
                net_memory_delta=0,
            )
            self._manager.shuffler.insert(
                py_split_and_pack(
                    table=chunk.table_view(),
                    splits=splits,
                    stream=chunk.stream,
                    br=br,
                    reservation=reservation,
                )
            )

        async def insert_index(
            self, chunk: TableChunk, partition_map: TableChunk
        ) -> None:
            """
            Partition chunk by a separate single-column partition-map and insert.

            Parameters
            ----------
            chunk
                The payload chunk to partition. Its schema is preserved
                unchanged in the shuffler output.
            partition_map
                Single-column ``TableChunk`` whose integer values give the
                target partition ID for each row. Must be row-aligned with
                ``chunk``.
            """
            br = self._manager.context.br()
            # As in `insert_hash_keys`, this covers the reorder plus the pack.
            reservation = await reserve_memory(
                self._manager.context,
                py_partition_and_pack_cost(chunk.table_view(), chunk.stream, br),
                # The chunk's data moves into shuffler-owned packed buffers,
                # nothing lasting is added.
                net_memory_delta=0,
            )
            reorder_nbytes = py_split_and_pack_cost(
                chunk.table_view(), chunk.stream, br
            )
            with stream_ordered_after(
                br.stream_pool.get_stream,
                upstreams=(chunk.stream, partition_map.stream),
            ) as stream:
                partition_map_col = partition_map.table_view().columns()[0]
                with opaque_memory_usage(reservation.split(reorder_nbytes)):
                    reordered, offsets = plc.partitioning.partition(
                        chunk.table_view(),
                        partition_map_col,
                        self._manager.num_partitions,
                        stream=stream,
                        mr=br.device_mr,
                    )
                self._manager.shuffler.insert(
                    py_split_and_pack(
                        table=reordered,
                        splits=list(offsets[1:-1]),
                        stream=stream,
                        br=br,
                        reservation=reservation,
                    )
                )

        async def __aenter__(self) -> ShuffleManager.Inserter:
            """Enter the context manager."""
            return self

        async def __aexit__(self, *args: object) -> None:
            """Exit the context manager, calling ``insert_finished()``."""
            await self._manager.shuffler.insert_finished(self._manager.context)

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        num_partitions: int,
        collective_id: int,
        *,
        partition_assignment: PartitionAssignment = PartitionAssignment.ROUND_ROBIN,
    ):
        self.context = context
        self.comm = comm
        self.num_partitions = num_partitions
        self.collective_id = collective_id
        self.shuffler = ShufflerAsync(
            context,
            comm,
            collective_id,
            num_partitions,
            partition_assignment=partition_assignment,
        )

    def inserting(self) -> ShuffleManager.Inserter:
        """Return a context manager for the insert phase."""
        return ShuffleManager.Inserter(self)

    def local_partitions(self) -> list[int]:
        """Get the local partition IDs for this rank."""
        return self.shuffler.local_partitions()

    async def extract_chunk(self, partition_id: int, stream: Stream) -> plc.Table:
        """
        Extract a chunk from the ShuffleManager.

        Parameters
        ----------
        partition_id
            The partition ID of the chunk to extract.
        stream
            The stream to use for chunk extraction.

        Returns
        -------
        The extracted table.
        """
        partitions = self.shuffler.extract(partition_id)
        reservation = await reserve_memory(
            self.context,
            py_unpack_and_concat_cost(partitions),
            # Representation change: the packed input is consumed as the
            # unpacked table is produced, at roughly the same size.
            net_memory_delta=0,
        )
        return py_unpack_and_concat(
            partitions=partitions,
            stream=stream,
            br=self.context.br(),
            reservation=reservation,
        )

    def extract_pieces(self, partition_id: int) -> list[PackedData]:
        """
        Extract raw packed items for a partition without unpacking.

        Parameters
        ----------
        partition_id
            The partition ID to extract.

        Returns
        -------
        list[PackedData]
            Raw packed items for the partition.
        """
        return self.shuffler.extract(partition_id)


class LocalRepartitioner:
    """
    Local re-partitioner that wraps a completed :class:`ShuffleManager`.

    Parameters
    ----------
    shuffle
        Completed inter-rank :class:`ShuffleManager` (insertion phase done).
        The repartitioner consumes whatever local partitions this rank owns.
    local_count
        Number of local output partitions to produce.
    """

    def __init__(self, shuffle: ShuffleManager, local_count: int) -> None:
        self._global_shuffle = shuffle
        self._br = shuffle.context.br()
        options = Options(get_environment_variables())
        local_comm = single_comm(options, shuffle.comm.progress_thread)
        local_ctx = Context(local_comm.logger, self._br, options)
        self._local_shuffle = ShuffleManager(
            local_ctx,
            local_comm,
            local_count,
            shuffle.collective_id,
        )

    def _iter_pieces(self) -> Iterator[Any]:
        for partition_id in self._global_shuffle.local_partitions():
            yield from self._global_shuffle.extract_pieces(partition_id)

    async def _unpack_table(self, piece: Any, stream: Stream) -> plc.Table | None:
        # TODO: batch pieces up to target_partition_size before unpacking
        pieces = [piece]
        reservation = await reserve_memory(
            self._global_shuffle.context,
            py_unpack_and_concat_cost(pieces),
            # Representation change: the packed input is consumed as the
            # unpacked table is produced, at roughly the same size.
            net_memory_delta=0,
        )
        table = py_unpack_and_concat(
            pieces, stream=stream, br=self._br, reservation=reservation
        )
        return None if table.num_rows() == 0 else table

    async def repartition_by_hash(
        self, *, columns_to_hash: tuple[int, ...], stream: Stream
    ) -> None:
        """
        Re-partition items by hash of the given columns.

        Parameters
        ----------
        columns_to_hash
            Tuple of column indices to use for hashing.
        stream
            CUDA stream for the unpack operation.
        """
        async with self._local_shuffle.inserting() as inserter:
            for piece in self._iter_pieces():
                table = await self._unpack_table(piece, stream)
                if table is not None:
                    chunk = TableChunk.from_pylibcudf_table(
                        table, stream, exclusive_view=True, br=self._br
                    )
                    del table
                    await inserter.insert_hash(chunk, columns_to_hash)
                    del chunk

    async def repartition_by_index(
        self,
        *,
        partition_col: int,
        stream: Stream,
        drop_partition_col: bool = True,
    ) -> None:
        """
        Re-partition items by a pre-computed integer column in the received data.

        Parameters
        ----------
        partition_col
            Index of the integer column whose values give the target
            local partition ID for each row.
        stream
            CUDA stream for the unpack operation.
        drop_partition_col
            If ``True`` (default), the partition column is stripped from the
            payload before inserting. If ``False``, it is kept in the output.
        """
        async with self._local_shuffle.inserting() as inserter:
            for piece in self._iter_pieces():
                table = await self._unpack_table(piece, stream)
                if table is None:
                    continue
                cols = table.columns()
                payload = TableChunk.from_pylibcudf_table(
                    plc.Table(
                        [
                            c
                            for i, c in enumerate(cols)
                            if not drop_partition_col or i != partition_col
                        ]
                    ),
                    stream,
                    exclusive_view=False,
                    br=self._br,
                )
                partition_map = TableChunk.from_pylibcudf_table(
                    plc.Table([cols[partition_col]]),
                    stream,
                    exclusive_view=False,
                    br=self._br,
                )
                await inserter.insert_index(payload, partition_map)
                del cols, table, payload, partition_map

    def local_partitions(self) -> list[int]:
        """Return the local partition IDs."""
        return self._local_shuffle.local_partitions()

    async def extract_chunk(self, partition_id: int, stream: Stream) -> plc.Table:
        """
        Extract the table for *partition_id* from the local shuffle.

        Parameters
        ----------
        partition_id
            The local partition to extract.
        stream
            CUDA stream for the unpack operation.
        """
        return await self._local_shuffle.extract_chunk(partition_id, stream)


def _key_column_indices(
    keys: tuple[NamedExpr, ...], schema: Schema
) -> tuple[int, ...] | None:
    """Return column indices for simple column keys, or None for expressions."""
    columns = list(schema)
    indices: list[int] = []
    for key in keys:
        if not isinstance(key.value, Col):
            return None
        indices.append(columns.index(key.value.name))
    return tuple(indices)


def _ensure_pointwise_keys(keys: tuple[NamedExpr, ...]) -> None:
    if not all(expr.is_pointwise for expr in traversal([key.value for key in keys])):
        raise NotImplementedError("Shuffle requires pointwise key expressions.")


# Smallest batch, in packed bytes, that `ShuffleManager.Inserter` gathers and
# packs at a time when the device memory available for reservation is lower.
_MIN_PACK_BATCH_NBYTES = 128 * 2**20

# Device bytes per row that `hash_partition_indices` with
# `PartitionOrder.UNSTABLE` allocates: the gather map, the partition numbers
# and, for up to about 1536 partitions, the per-tile partition histograms.
_HASH_PARTITION_INDICES_NBYTES_PER_ROW = 10


def _hash_partition_indices_cost(num_rows: int) -> int:
    """Device memory in bytes for `hash_partition_indices` over ``num_rows`` rows."""
    return _HASH_PARTITION_INDICES_NBYTES_PER_ROW * num_rows


def _device_memory_available(br: BufferResource) -> int:
    """Device memory in bytes that ``br`` can reserve without spilling."""
    return br.memory_available_for_reservation(MemoryType.DEVICE)


def _batch_end(offsets: list[int], begin: int, max_rows: int) -> int:
    """
    End of the batch of consecutive partitions that starts at partition ``begin``.

    The batch holds as many partitions as fit in ``max_rows`` rows, and at
    least one.

    Parameters
    ----------
    offsets
        ``num_partitions + 1`` row offsets, where partition ``i`` has
        ``offsets[i + 1] - offsets[i]`` rows.
    begin
        The first partition of the batch.
    max_rows
        The maximum number of rows of a batch of more than one partition.

    Returns
    -------
    One past the last partition of the batch.
    """
    num_partitions = len(offsets) - 1
    end = begin + 1
    while end < num_partitions and offsets[end + 1] - offsets[begin] <= max_rows:
        end += 1
    return end


def _evaluate_key_table(
    chunk: TableChunk, keys: tuple[NamedExpr, ...], schema: Schema
) -> plc.Table:
    df = DataFrame.from_table(
        chunk.table_view(),
        list(schema.keys()),
        list(schema.values()),
        chunk.stream,
    )
    return plc.Table([key.evaluate(df).obj for key in keys])


async def _global_shuffle(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    keys_to_hash: tuple[NamedExpr, ...],
    input_schema: Schema,
    num_partitions: int,
    collective_id: int,
) -> None:
    """
    Global shuffle implementation.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    ir_context
        The execution context for the IR node.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    keys_to_hash
        Tuple of expressions to use for hashing.
    input_schema
        Schema of incoming chunks used to evaluate ``keys_to_hash``.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    """
    _ensure_pointwise_keys(keys_to_hash)
    columns_to_hash = _key_column_indices(keys_to_hash, input_schema)
    metadata_in = await recv_metadata(ch_in, context)

    # Check if we can skip the shuffle (already partitioned correctly)
    if columns_to_hash is not None and _is_already_partitioned(
        metadata_in, columns_to_hash, num_partitions, comm.nranks
    ):
        # Forward metadata and data unchanged
        await send_metadata(ch_out, context, metadata_in)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)
        return

    # Normal shuffle path
    output_metadata = ChannelMetadata(
        local_count=max(1, num_partitions // comm.nranks),
        partitioning=(
            Partitioning(
                inter_rank=HashScheme(columns_to_hash, num_partitions),
                local="inherit",
            )
            if columns_to_hash is not None
            else None
        ),
    )
    await send_metadata(ch_out, context, output_metadata)

    # When input is duplicated, only rank 0 should contribute data.
    # Other ranks still participate in the shuffle protocol.
    skip_insert = metadata_in.duplicated and comm.rank != 0

    shuffle = ShuffleManager(context, comm, num_partitions, collective_id)
    async with shuffle.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            if not skip_insert:
                chunk, _ = await make_table_chunks_available_or_wait(
                    context,
                    TableChunk.from_message(msg, br=context.br()),
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                if columns_to_hash is None:
                    await inserter.insert_hash_keys(chunk, keys_to_hash, input_schema)
                else:
                    await inserter.insert_hash(
                        chunk,
                        columns_to_hash,
                    )

    for partition_id in shuffle.local_partitions():
        stream = ir_context.get_cuda_stream()
        await ch_out.send(
            context,
            Message(
                partition_id,
                TableChunk.from_pylibcudf_table(
                    table=await shuffle.extract_chunk(partition_id, stream),
                    stream=stream,
                    exclusive_view=True,
                    br=context.br(),
                ),
            ),
        )

    await ch_out.drain(context)


@define_actor()
async def shuffle_actor(
    context: Context,
    comm: Communicator,
    ir: Shuffle,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    keys_to_hash: tuple[NamedExpr, ...],
    input_schema: Schema,
    num_partitions: int,
    collective_id: int,
) -> None:
    """
    Execute a global shuffle pipeline within a single node.

    This node combines partition_and_pack, shuffler, and unpack_and_concat
    into a single Python node using rapidsmpf.shuffler.Shuffler and utilities
    from cudf_streaming.partition_utils.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Shuffle IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    keys_to_hash
        Tuple of expressions to use for hashing.
    input_schema
        Schema of incoming chunks used to evaluate ``keys_to_hash``.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        await _global_shuffle(
            context,
            comm,
            ir_context,
            ch_out,
            ch_in,
            keys_to_hash,
            input_schema,
            num_partitions,
            collective_id,
        )


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Local shuffle operation.

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    _ensure_pointwise_keys(ir.keys)

    context = rec.state["context"]
    num_partitions = rec.state["partition_info"][ir].count

    # Look up the reserved collective ID for this operation
    collective_id = rec.state["collective_id_map"][ir][0]

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])
    ir_context = ir_context_for_node(rec, ir)

    # Complete shuffle node
    nodes[ir] = [
        shuffle_actor(
            context,
            rec.state["comm"],
            ir,
            ir_context,
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            keys_to_hash=ir.keys,
            input_schema=child.schema,
            num_partitions=num_partitions,
            collective_id=collective_id,
        )
    ]

    return nodes, channels
