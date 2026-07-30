#!/usr/bin/env bash
set -Eeuo pipefail

# The checked-in bindings and Rust runtime use this exact FlatBuffers release.
readonly FLATC_VERSION='25.12.19'
readonly FLATC_ARCHIVE='Linux.flatc.binary.g++-13.zip'
readonly FLATC_SHA256='9f87066dc5dfa7fe02090b55bab5f3e55df03e32c9b0cdf229004ade7d091039'
readonly FLATC_URL="https://github.com/google/flatbuffers/releases/download/v${FLATC_VERSION}/Linux.flatc.binary.g%2B%2B-13.zip"

destination="${1:-${RUNNER_TEMP:-/tmp}/flatc-${FLATC_VERSION}}"
archive="${destination}.zip"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ;;
  *) echo 'The pinned flatc binary supports Linux x86_64 only.' >&2; exit 2 ;;
esac

mkdir -p "$destination"
curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
  --connect-timeout 20 --retry 5 --retry-delay 2 --retry-all-errors \
  --output "$archive" "$FLATC_URL"
printf '%s  %s\n' "$FLATC_SHA256" "$archive" | sha256sum --check --status
unzip -q -o "$archive" -d "$destination"
rm -f "$archive"

flatc_path="$(find "$destination" -type f -name flatc -print -quit)"
if [[ -z "$flatc_path" ]]; then
  echo "${FLATC_ARCHIVE} did not contain flatc" >&2
  exit 1
fi
chmod 0755 "$flatc_path"
"$flatc_path" --version >&2
printf '%s\n' "$flatc_path"
