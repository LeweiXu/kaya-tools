#!/usr/bin/env bash
# Put `kaya` on your PATH (a symlink in ~/.local/bin) and create config.toml.
# Nothing else is installed: kaya is plain Python 3.11+ with no dependencies.
# Safe to rerun.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${KAYA_PYTHON:-python3}"
bindir="${KAYA_BIN_DIR:-$HOME/.local/bin}"

if ! "$py" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "Need Python 3.11 or newer ('$py' is older or missing). Set KAYA_PYTHON=/path/to/python3.11+." >&2
    exit 1
fi
for tool in ssh rsync; do
    command -v "$tool" >/dev/null || echo "warning: $tool is not installed; kaya needs it" >&2
done

chmod +x "$root/bin/kaya"
mkdir -p "$bindir"
link="$bindir/kaya"
if [ -e "$link" ] || [ -L "$link" ]; then
    if [ "$(readlink -f "$link")" != "$root/bin/kaya" ]; then
        echo "$link already exists and points somewhere else; move it and rerun." >&2
        exit 1
    fi
else
    ln -s "$root/bin/kaya" "$link"
    echo "Linked $link -> $root/bin/kaya"
fi

if [ ! -f "$root/config.toml" ]; then
    cp "$root/config.example.toml" "$root/config.toml"
    echo "Created $root/config.toml: fill in [ssh] and [cluster]."
fi

case ":$PATH:" in
    *":$bindir:"*) echo "Done. Try: kaya doctor" ;;
    *) echo "Done, but $bindir is not on your PATH yet. Most distros add it at login if it exists," \
            "so open a new terminal (or log out and in), then try: kaya doctor" ;;
esac
