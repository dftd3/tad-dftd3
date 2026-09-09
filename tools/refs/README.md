# tools/refs

Regenerates `test/test_model/samples.py`'s coordination number, reference
weights and C6 fixture data. These three quantities have no accessor in the
`dftd3` Python package's C API, so getting them at all means linking
s-dftd3 as a Fortran library instead -- a heavier, from-source dependency
that only this regeneration step needs, not the test suite itself.

`build.sh` fetches s-dftd3 v1.6.0 from GitHub and builds it from source,
via [fpm](https://fpm.fortran-lang.org/) (`fpm.toml`) or
[Meson](https://mesonbuild.com/) (`meson.build`, `subprojects/s-dftd3.wrap`)
-- whichever is on `PATH` (fpm preferred; pass `meson` to force one). No
compiler is hardcoded: both build systems use whatever Fortran compiler
they default to, or `$FC`/`FPM_FC` if set.

```sh
tools/refs/build.sh
python tools/refs/gen_samples.py > /tmp/refs.py
```

Then paste the printed `refs: Dict[str, Refs] = {...}` block into
`test/test_model/samples.py`, replacing the old one.
