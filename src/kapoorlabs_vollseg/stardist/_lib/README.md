# Vendored stardist C++ implementation

These files are copied verbatim from the upstream [stardist] project
(commit at copy time matched ``/mnt/disk/python_workspace/stardist``)
and re-distributed under their original BSD-3 license — see
``STARDIST_LICENSE.txt`` in this folder.

We keep them here for two reasons:

1. **Algorithm source-of-truth.** Our Python port of
   ``_COMMON_polyhedron_to_label`` / ``c_non_max_suppression_inds`` lives
   in ``../inference.py``. When something doesn't behave the same way as
   upstream stardist, we diff against ``stardist3d_impl.cpp`` here
   instead of jumping repos.

2. **Future native compilation.** ``stardist3d_impl.cpp`` is the C++
   kernel originally compiled as a CPython extension via
   ``stardist3d.cpp`` (the Cython binding). Pure-numpy ports of the
   tetrahedron rasterizer and the NMS overlap test work, but they're
   ~50–100× slower than the OpenMP C++ kernel. If raster throughput
   ever becomes the bottleneck, the path to native speed is:

   ```
   # vendor the headers (NOT copied here — see below)
   mkdir external && cd external
   git clone --depth 1 https://github.com/qhull/qhull qhull_src
   git clone --depth 1 https://github.com/jlblancoc/nanoflann nanoflann

   # then build the .so as upstream stardist does
   make -C src/kapoorlabs_vollseg/stardist/_lib lib
   ```

   ``external/qhull_src/`` and ``external/nanoflann/`` are NOT vendored
   here — they're 50 kLOC of third-party C with their own licenses and
   no point bloating our repo unless someone actually builds.

## Files

| File                     | Purpose                                              |
|--------------------------|------------------------------------------------------|
| ``stardist3d_impl.cpp``  | Core C++ algorithms (NMS, polyhedron_to_label, etc.) |
| ``stardist3d_impl.h``    | Public API of the impl module                        |
| ``stardist3d_lib.c/h``   | C ABI shim used by the Fiji plugin / shared library  |
| ``stardist3d.cpp``       | CPython extension entry points                       |
| ``utils.cpp/h``          | Misc helpers (signal handler, timing)                |
| ``test_lib3d.cpp``       | Smoke test for the C++ NMS path                      |
| ``Makefile``             | Upstream build script (linux + osx, requires gcc-11) |

## License

Upstream stardist is BSD-3 — see ``STARDIST_LICENSE.txt``. The Python
port in ``../inference.py`` is a clean-room implementation by us; the
files in this directory are direct copies and carry stardist's
copyright + license.

[stardist]: https://github.com/stardist/stardist
