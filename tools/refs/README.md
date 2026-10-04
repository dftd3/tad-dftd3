# tools/refs

Regenerates `test/references/`, one JSON file per molecule holding the
coordination number, reference weights and C6 fixture data. These three
quantities have no accessor in the `dftd3` Python package's C API, so
getting them at all means linking s-dftd3 as a Fortran library instead --
a heavier, from-source dependency that only this regeneration step needs,
not the test suite itself.

Building `gen_refs_fortran` fetches s-dftd3 v1.6.0 from GitHub and builds
it from source, via [fpm](https://fpm.fortran-lang.org/) (`fpm.toml`) or
[Meson](https://mesonbuild.com/) (`meson.build`, `subprojects/s-dftd3.wrap`)
-- whichever you prefer. Neither hardcodes a compiler: both use whatever
Fortran compiler they default to, or `$FC`/`FPM_FC` if set.

With fpm:

```sh
cd tools/refs
fpm install --profile release --build-dir _build_fpm --prefix _install_fpm
```

With Meson:

```sh
cd tools/refs
meson setup _build_meson --buildtype=release --prefix "$PWD/_install_meson"
meson install -C _build_meson
```

Either way, `gen_references.py` finds the installed binary itself, at
`_install_fpm/bin/gen_refs_fortran` or `_install_meson/bin/gen_refs_fortran`
respectively (both -- along with the `_build_fpm`/`_build_meson` build
directories -- gitignored: machine-specific, not portable; only the
Fortran source is tracked). Then:

```sh
python tools/refs/gen_refs.py
```

`gen_refs.py` writes one `test/references/<molecule>.json` per entry
in its `SAMPLE_LIST`, in place; commit the result.
