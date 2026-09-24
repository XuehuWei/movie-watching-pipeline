#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_dir="$repo_dir/_third_party/opensmile"

if [[ -x "$target_dir/build/progsrc/smilextract/SMILExtract" ]]; then
  echo "openSMILE is already installed at: $target_dir"
  exit 0
fi

for command in git cmake; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Error: $command is required to build openSMILE." >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$target_dir")"
if [[ ! -d "$target_dir/.git" ]]; then
  git clone --depth 1 https://github.com/audeering/opensmile.git "$target_dir"
fi

(
  cd "$target_dir"
  bash build.sh
)

binary="$target_dir/build/progsrc/smilextract/SMILExtract"
config="$target_dir/config/egemaps/v01a/eGeMAPSv01a.conf"
if [[ ! -x "$binary" || ! -f "$config" ]]; then
  echo "Error: openSMILE build completed but required files were not found." >&2
  exit 1
fi

echo "openSMILE installed successfully at: $target_dir"
