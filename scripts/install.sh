#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
project_root="$(pwd -P)"
umask 077
stage_root="$(realpath -m -- "$project_root/.shum-dist-staging-$$-$RANDOM-$RANDOM")"
[[ "$stage_root" == "$project_root"/.shum-dist-staging-* ]] || {
  echo "Invalid staging directory" >&2
  exit 1
}
mkdir -- "$stage_root"
cleanup() {
  if [[ "$stage_root" == "$project_root"/.shum-dist-staging-* &&
        -d "$stage_root" && ! -L "$stage_root" ]]; then
    rm -rf -- "$stage_root"
  fi
}
trap cleanup EXIT
scripts/build-dist.sh "$stage_root/release"
"$stage_root/release/bin/shum" --onboarding "$@"
