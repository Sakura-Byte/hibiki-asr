#!/usr/bin/env bash
# Install hibiki-asr on Linux: uv (if it is missing), the engine as a uv tool, and the runtime that fits this
# machine's hardware. Needs no sudo, and is safe to run again (it reinstalls the engine and re-applies the runtime).
#
#   curl -LsSf https://raw.githubusercontent.com/Sakura-Byte/hibiki-asr/main/install.sh | bash
#
# Environment:
#   HIBIKI_ASR_REF              branch, tag or commit to install (default: the repository's default branch)
#   HIBIKI_ASR_INSTALL_VARIANT  runtime variant for `hibiki-asr setup` (default: auto, which follows the hardware)
set -euo pipefail

REPO_URL="https://github.com/Sakura-Byte/hibiki-asr"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"
REF="${HIBIKI_ASR_REF:-}"
VARIANT="${HIBIKI_ASR_INSTALL_VARIANT:-auto}"
# These belong to this script; the engine must not see them (HIBIKI_ASR_* variables configure it).
unset HIBIKI_ASR_REF HIBIKI_ASR_INSTALL_VARIANT

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "this installer supports Linux only. On other systems: uv tool install \"hibiki-asr[runtime] @ git+${REPO_URL}\" && hibiki-asr setup"

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (https://docs.astral.sh/uv/) ..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$UV_INSTALLER_URL" | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$UV_INSTALLER_URL" | sh
  else
    die "curl or wget is needed to download uv"
  fi
  # uv's installer puts it in ~/.local/bin, which this shell does not know about yet
  export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv was installed but is not on PATH; open a new shell and run this script again"
fi

# [runtime] is the CPU baseline, so the engine works even if the GPU step below fails; `setup` then swaps in the
# pinned runtime for this machine.
SPEC="hibiki-asr[runtime] @ git+${REPO_URL}${REF:+@${REF}}"
say "Installing hibiki-asr (${REF:-default branch}) ..."
uv tool install --force "$SPEC"

TOOL_BIN="$(uv tool dir --bin)"
export PATH="${TOOL_BIN}:$PATH"
command -v hibiki-asr >/dev/null 2>&1 || die "hibiki-asr was installed but is not in ${TOOL_BIN}"

say ""
say "Choosing the runtime for this machine ..."
if ! hibiki-asr setup --variant "$VARIANT" --yes </dev/null; then
  say ""
  say "hibiki-asr is installed and runs on the CPU, but the runtime step reported a problem (see above)."
  say "Fix it and run:  hibiki-asr setup --variant ${VARIANT}"
  exit 1
fi

cat <<NEXT

hibiki-asr is installed.

Next steps:
  hibiki-asr models sources                    # can Hugging Face be reached? which mirror is fastest?
  hibiki-asr models download chickenrice@v2    # the translate model, about 3 GB, with the VAD it needs
  hibiki-asr serve                             # listens on http://127.0.0.1:8001

Optional:
  hibiki-asr service install                   # start it automatically when you log in
  hibiki-asr update                            # upgrade later

If \`hibiki-asr\` is not found in a new shell, run:  uv tool update-shell
NEXT
