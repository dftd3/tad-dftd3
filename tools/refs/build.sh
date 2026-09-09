#!/usr/bin/env bash
# This file is part of tad-dftd3.
# SPDX-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build tools/refs/dump_reference, fetching s-dftd3 v1.6.0 from GitHub and
# building it from source -- via fpm (fpm.toml) or Meson (meson.build,
# subprojects/s-dftd3.wrap), whichever is available. See README.md in this
# directory. Neither path hardcodes a compiler: fpm defaults to gfortran
# unless FPM_FC/--compiler says otherwise, and Meson uses $FC or whatever
# its own default is.
#
# Usage:
#     tools/refs/build.sh [fpm|meson]   # default: fpm, falling back to meson

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${1:-}"

if [[ -z "$BACKEND" ]]; then
    if command -v fpm >/dev/null; then
        BACKEND=fpm
    elif command -v meson >/dev/null && command -v ninja >/dev/null; then
        BACKEND=meson
    else
        echo "error: neither fpm nor meson+ninja found on PATH" >&2
        exit 1
    fi
fi

case "$BACKEND" in
fpm)
    ( cd "$HERE" && fpm install --profile release --prefix "$HERE/_fpm-install" )
    cp "$HERE/_fpm-install/bin/dump_reference" "$HERE/dump_reference"
    ;;
meson)
    BUILD_DIR="$HERE/builddir"
    if [[ ! -d "$BUILD_DIR" ]]; then
        meson setup "$BUILD_DIR" "$HERE" --buildtype=release
    fi
    ninja -C "$BUILD_DIR" dump_reference
    cp "$BUILD_DIR/dump_reference" "$HERE/dump_reference"
    ;;
*)
    echo "error: unknown backend '$BACKEND' (expected 'fpm' or 'meson')" >&2
    exit 1
    ;;
esac

echo "built $HERE/dump_reference (with $BACKEND)"
