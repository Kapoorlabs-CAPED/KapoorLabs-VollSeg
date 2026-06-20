"""Tile iterator ported verbatim from CSBDeep / StarDist.

This is a line-by-line port of ``csbdeep.internals.predict.Tile``,
``Tiling``, and ``tile_iterator_1d`` — so that StarDist inference here
follows the **exact** stitching logic the upstream model uses
(``predict_instances`` → ``tile_iterator``). No blending, no weighted
overlap: each output voxel is written exactly once from exactly one
tile, with boundary tiles claiming the boundary region instead of
having their weight ramp to zero.

The yield tuple from :func:`tile_iterator` is
``(tile_array, crop_slices, write_slices)`` matching CSBDeep's contract:

* ``tile_array`` — the input slab to feed the network (with overlap).
* ``crop_slices`` — what region of the network's output for this tile
  to copy out (drops the overlap region, except at the volume boundary
  where the tile owns the boundary).
* ``write_slices`` — where to write that cropped output in the
  destination buffer.

So the consumer loop is::

    for tile, crop, write in tile_iterator(x, n_tiles, block_sizes, n_block_overlaps):
        pred = model(tile)
        dst[write] = pred[crop]

— same shape as the original ``model.base._predict_generator``.
"""

from __future__ import annotations

import numpy as np


def _raise(e):
    raise e


class Tile:
    """Single tile along one axis — direct port of
    :class:`csbdeep.internals.predict.Tile`."""

    def __init__(self, n, size, overlap, prev):
        self.n = int(n)
        self.size = int(size)
        self.overlap = int(overlap)
        if self.n < self.size:
            assert prev is None
            # print("Truncating tile size from %d to %d." % (self.size, self.n))
            self.size = self.n
            self.overlap = 0
        assert self.size > 2 * self.overlap
        # assert self.n >= self.size
        if prev is not None:
            assert not prev.at_end, "Previous tile already at end"
        self.prev = prev
        self.read_slice = self._read_slice
        self.write_slice = self._write_slice

    @property
    def at_begin(self):
        return self.prev is None

    @property
    def at_end(self):
        return self.read_slice.stop == self.n

    @property
    def _read_slice(self):
        if self.at_begin:
            start, stop = 0, self.size
        else:
            prev_read_slice = self.prev.read_slice
            start = prev_read_slice.stop - 2 * self.overlap
            stop = start + self.size
            shift = min(0, self.n - stop)
            start, stop = start + shift, stop + shift
            assert start > prev_read_slice.start
        assert start >= 0 and stop <= self.n
        return slice(start, stop)

    @property
    def _write_slice(self):
        if self.at_begin:
            if self.at_end:
                return slice(0, self.n)
            else:
                return slice(0, self.size - 1 * self.overlap)
        elif self.at_end:
            s = self.prev.write_slice.stop
            return slice(s, self.n)
        else:
            s = self.prev.write_slice.stop
            return slice(s, s + self.size - 2 * self.overlap)

    def __repr__(self):
        s = np.array(list(" " * self.n))
        s[self.read_slice] = "-"
        s[self.write_slice] = "x" if (self.at_begin or self.at_end) else "o"
        return "".join(s)


class Tiling:
    """Sequence of :class:`Tile` covering one axis. Direct port of
    :class:`csbdeep.internals.predict.Tiling`."""

    def __init__(self, n, size, overlap):
        self.n = n
        self.size = size
        self.overlap = overlap
        tiles = [Tile(prev=None, **self.__dict__)]
        while not tiles[-1].at_end:
            tiles.append(Tile(prev=tiles[-1], **self.__dict__))
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __repr__(self):
        return "\n".join(f"{i:3}. {t}" for i, t in enumerate(self.tiles, 1))

    def slice_generator(self, block_size=1):
        def scale(sl):
            return slice(block_size * sl.start, block_size * sl.stop)

        def crop_slice(read, write):
            stop = write.stop - read.stop
            return slice(write.start - read.start, stop if stop < 0 else None)

        for t in self.tiles:
            read, write = scale(t.read_slice), scale(t.write_slice)
            yield read, write, crop_slice(read, write)

    @staticmethod
    def for_n_tiles(n, n_tiles, overlap):
        smallest_size = 2 * overlap + 1
        tile_size = smallest_size  # start with smallest posible tile_size
        while len(Tiling(n, tile_size, overlap)) > n_tiles:
            tile_size += 1
        if tile_size == smallest_size:
            return Tiling(n, tile_size, overlap)
        candidates = (
            Tiling(n, tile_size - 1, overlap),
            Tiling(n, tile_size, overlap),
        )
        diffs = [np.abs(len(c) - n_tiles) for c in candidates]
        return candidates[int(np.argmin(diffs))]


def total_n_tiles(x, n_tiles, block_sizes, n_block_overlaps):
    """Total number of tiles the iterator will yield. Direct port of
    :func:`csbdeep.internals.predict.total_n_tiles` (``guarantee='size'``
    only — the path used by stardist's ``predict_instances``)."""
    assert x.ndim == len(n_tiles) == len(block_sizes) == len(n_block_overlaps)
    n_tiles_used = 1
    for n, n_tile, block_size, n_block_overlap in zip(
        x.shape, n_tiles, block_sizes, n_block_overlaps
    ):
        assert n % block_size == 0
        n_blocks = n // block_size
        n_tiles_used *= len(Tiling.for_n_tiles(n_blocks, n_tile, n_block_overlap))
    return n_tiles_used


def tile_iterator_1d(x, axis, n_tiles, block_size, n_block_overlap):
    """One-axis tile iterator. Direct port of
    :func:`csbdeep.internals.predict.tile_iterator_1d` (``guarantee='size'``)."""
    n = x.shape[axis]
    n % block_size == 0 or _raise(
        ValueError("'x' must be evenly divisible by 'block_size' along 'axis'")
    )
    n_blocks = n // block_size

    tiling = Tiling.for_n_tiles(n_blocks, n_tiles, n_block_overlap)

    def ndim_slices(t):
        sl = [slice(None)] * x.ndim
        sl[axis] = t
        return tuple(sl)

    for read, write, crop in tiling.slice_generator(block_size):
        tile_in = read  # src in input image     / tile
        tile_out = write  # dst in output image   / s_dst
        tile_crop = crop  # crop of src for output / s_src
        yield x[ndim_slices(tile_in)], ndim_slices(tile_crop), ndim_slices(tile_out)


def tile_iterator(x, n_tiles, block_sizes, n_block_overlaps):
    """N-d tile iterator. Direct port of
    :func:`csbdeep.internals.predict.tile_iterator` (``guarantee='size'``).

    Yields ``(tile, crop_slices, write_slices)`` triples where ``tile``
    is a view into ``x`` (with overlap), ``crop_slices`` selects the
    region of the tile's prediction to copy, and ``write_slices`` is
    where to write that cropped region in the destination buffer.
    """
    if np.isscalar(n_tiles):
        n_tiles = (n_tiles,) * x.ndim
    if np.isscalar(block_sizes):
        block_sizes = (block_sizes,) * x.ndim
    if np.isscalar(n_block_overlaps):
        n_block_overlaps = (n_block_overlaps,) * x.ndim

    if not (x.ndim == len(n_tiles) == len(block_sizes) == len(n_block_overlaps)):
        raise ValueError(
            "x.ndim, n_tiles, block_sizes, n_block_overlaps must agree "
            f"({x.ndim}, {n_tiles}, {block_sizes}, {n_block_overlaps})"
        )

    def _accumulate(x, axis, n_tiles, block_size, n_block_overlap):
        if n_tiles == 1:
            full_in = tuple(slice(0, s) for s in x.shape)
            full_crop = tuple(slice(None) for _ in x.shape)
            yield x, full_crop, full_in
        else:
            yield from tile_iterator_1d(x, axis, n_tiles, block_size, n_block_overlap)

    def _recurse(x, axis, write_prefix, crop_prefix):
        if axis == x.ndim:
            yield x, tuple(crop_prefix), tuple(write_prefix)
            return
        for sub, crop_sl, write_sl in _accumulate(
            x, axis, n_tiles[axis], block_sizes[axis], n_block_overlaps[axis]
        ):
            # ``crop_sl`` and ``write_sl`` are full-ndim slice tuples
            # along ``axis`` only; merge with the prefix.
            new_write = list(write_prefix)
            new_crop = list(crop_prefix)
            new_write[axis] = write_sl[axis]
            new_crop[axis] = crop_sl[axis]
            yield from _recurse(sub, axis + 1, new_write, new_crop)

    init_write = [slice(0, s) for s in x.shape]
    init_crop = [slice(None) for _ in x.shape]
    yield from _recurse(x, 0, init_write, init_crop)


__all__ = ["Tile", "Tiling", "total_n_tiles", "tile_iterator"]
