#!/usr/bin/env bash
# Offline, non-interactive Linux installation smoke test. Only touches a private
# directory next to this checkout; never uses the caller's HOME or XDG directories.
set -euo pipefail

usage() {
  echo "Usage: scripts/onboarding-smoke.sh [--binary /absolute/path/to/shum]" >&2
  exit 2
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
binary="$root/target/debug/shum"
if (( $# )); then
  [[ $# == 2 && $1 == --binary && $2 == /* ]] || usage
  binary="$2"
fi
[[ -f "$binary" && -x "$binary" ]] || { echo "Build shum first or pass --binary /absolute/path" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 2; }

umask 077
# mkdir is atomic. Installation inside the source checkout is intentionally
# rejected, so use its parent rather than the real HOME or system temp.
scratch_parent="$(dirname "$root")"
scratch="$scratch_parent/.onboarding-smoke-$$-$RANDOM-$RANDOM"
mkdir -m 700 -- "$scratch"
cleanup() {
  if [[ "$scratch" == "$scratch_parent"/.onboarding-smoke-* && -d "$scratch" && ! -L "$scratch" ]]; then
    rm -rf -- "$scratch"
  fi
}
trap cleanup EXIT
export HOME="$scratch/home" XDG_CONFIG_HOME="$scratch/xdg"
mkdir -m 700 -- "$HOME" "$XDG_CONFIG_HOME"

install="$scratch/Install with spaces ü"
bin="$scratch/programs"
settings="$scratch/settings"
projects="$scratch/projects"
models="$scratch/models"
settings_file="$settings/settings.toml"
output="$scratch/output"
run() {
  timeout 30 "$@" </dev/null >"$output" 2>&1 || {
    cat "$output" >&2
    echo "Unexpected failure (or timeout): $*" >&2
    exit 1
  }
}
reject() {
  if timeout 30 "$@" </dev/null >"$output" 2>&1; then
    echo "Unexpected success: $*" >&2
    exit 1
  else
    status=$?
  fi
  if [[ $status == 124 ]]; then
    echo "Onboarding timed out: $*" >&2
    exit 1
  fi
}
contains() {
  grep -Fq -- "$1" "$2" || { echo "Expected '$1' in $2" >&2; cat "$2" >&2; exit 1; }
}
same() {
  cmp -s -- "$1" "$2" || { echo "Changed unexpectedly: $1" >&2; exit 1; }
}

# An unconfigured non-interactive call must fail promptly with a recovery step.
run "$binary" --help
reject env SHUM_SETTINGS="$settings_file" "$binary"
contains "Run \`shum --onboarding --storage-dir PATH\`" "$output"
printf 'invalid = [\n' >"$scratch/broken-settings.toml"
reject env SHUM_SETTINGS="$scratch/broken-settings.toml" "$binary" --onboarding \
  --storage-dir "$scratch/should-not-install"
contains "invalid settings" "$output"
contains "invalid = [" "$scratch/broken-settings.toml"
reject env SHUM_SETTINGS="$scratch/source-settings.toml" "$binary" --onboarding \
  --storage-dir "$root/.should-not-install"
contains "inside the Shum source checkout" "$output"
[[ ! -e "$root/.should-not-install" ]] || { echo "Wrote into source checkout" >&2; exit 1; }

# Separate custom locations, Unicode/space handling, and no network/model setup.
run env SHUM_SETTINGS="$settings_file" "$binary" --onboarding \
  --storage-dir "$install" --bin-dir "$bin" --settings-dir "$settings" \
  --projects-dir "$projects" --models-dir "$models"
contains "Shum: $bin/shum" "$output"
contains "Projects: $projects" "$output"
contains "Models: $models" "$output"
contains "Settings: $settings/config.toml" "$output"
[[ -x "$bin/shum" && -f "$bin/shum.installation.toml" && -f "$settings_file" ]] ||
  { echo "Installation files are missing" >&2; exit 1; }
bundled_asr=("$install"/runtime/*/adapters/asr/main.py)
[[ ${#bundled_asr[@]} == 1 && -f ${bundled_asr[0]} ]] ||
  { echo "Bundled runtime is missing" >&2; exit 1; }
contains 'enabled = false' "$settings/config.toml"
contains "settings = \"$settings_file\"" "$bin/shum.installation.toml"
contains "installation_dir = \"$install\"" "$settings_file"
contains "models_dir = \"$models\"" "$settings_file"
[[ ! -e "$models/primeline" && ! -e "$install/runtime/environments" ]] ||
  { echo "The offline onboarding path unexpectedly installed models or environments" >&2; exit 1; }
[[ ! -e "$HOME/.local/bin/shum" && ! -e "$HOME/bin/shum" &&
   ! -e "$HOME/.bashrc" && ! -e "$HOME/.zshrc" ]] ||
  { echo "Headless layout setup unexpectedly registered a command or modified shell files" >&2; exit 1; }
run "$bin/shum" --installation "$settings_file"
contains "Shum is configured. Projects: $projects" "$output"

# Register explicitly under a PATH directory. A different working directory
# must not affect command lookup; repeated setup must preserve the same link.
command_dir="$scratch/commands-on-path"
mkdir -- "$command_dir"
lookup_path="$command_dir:$PATH"
run env PATH="$lookup_path" "$bin/shum" --installation "$settings_file" \
  setup-command --command-dir "$command_dir"
[[ -L "$command_dir/shum" && $(readlink -f -- "$command_dir/shum") == "$bin/shum" ]] ||
  { echo "Command registration did not link to the installed Shum binary" >&2; exit 1; }
link_inode="$(stat -c %i -- "$command_dir/shum")"
run env PATH="$lookup_path" bash -c \
  'cd "$1"; [[ $(command -v shum) == "$2" ]] && shum --help' \
  _ "$scratch" "$command_dir/shum"
contains "Usage:" "$output"
run env PATH="$lookup_path" "$bin/shum" --installation "$settings_file" \
  setup-command --command-dir "$command_dir"
[[ $(stat -c %i -- "$command_dir/shum") == "$link_inode" ]] ||
  { echo "Repeated command registration replaced an existing link" >&2; exit 1; }
foreign_dir="$scratch/foreign-command"
mkdir -- "$foreign_dir"
printf 'foreign executable\n' >"$foreign_dir/shum"
reject "$bin/shum" --installation "$settings_file" setup-command \
  --command-dir "$foreign_dir"
contains "foreign executable" "$foreign_dir/shum"

# Shell setup is an explicit opt-in and affects only this test's private HOME.
# A newly started interactive Bash must resolve shum without inheriting the
# new command directory in its PATH.
shell_dir="$scratch/shell-commands"
mkdir -- "$shell_dir"
printf '# private bashrc\n' >"$HOME/.bashrc"
printf '# private bash profile\n' >"$HOME/.bash_profile"
run env SHELL=/bin/bash "$bin/shum" --installation "$settings_file" \
  setup-command --command-dir "$shell_dir" --setup-shell
contains '# private bashrc' "$HOME/.bashrc"
contains '# private bash profile' "$HOME/.bash_profile"
[[ -L "$shell_dir/shum" && $(readlink -f -- "$shell_dir/shum") == "$bin/shum" ]] ||
  { echo "Shell setup did not register the installed binary" >&2; exit 1; }
run env PATH="$PATH" bash --noprofile --rcfile "$HOME/.bashrc" -ic \
  '[[ $(command -v shum) == "$1" ]] && shum --help' _ "$shell_dir/shum"
contains "Usage:" "$output"
cp -- "$HOME/.bashrc" "$scratch/bashrc-after-setup"
cp -- "$HOME/.bash_profile" "$scratch/bash-profile-after-setup"
run env SHELL=/bin/bash "$bin/shum" --installation "$settings_file" \
  setup-command --command-dir "$shell_dir" --setup-shell
same "$scratch/bashrc-after-setup" "$HOME/.bashrc"
same "$scratch/bash-profile-after-setup" "$HOME/.bash_profile"

# The API's fully explicit, environment-free layout must not import or
# redirect an existing global installation, even when launched by its binary.
cp -- "$XDG_CONFIG_HOME/shum/installation.toml" "$scratch/old-global-locator"
isolated="$scratch/isolated"
onboarding_command_dir="$scratch/onboarding-command"
mkdir -- "$onboarding_command_dir"
run env PATH="$onboarding_command_dir:$PATH" "$bin/shum" --onboarding \
  --storage-dir "$isolated" --bin-dir "$scratch/isolated-bin" \
  --settings-dir "$scratch/isolated-settings" \
  --projects-dir "$scratch/isolated-projects" --models-dir "$scratch/isolated-models" \
  --command-dir "$onboarding_command_dir"
[[ -L "$onboarding_command_dir/shum" &&
   $(readlink -f -- "$onboarding_command_dir/shum") == "$scratch/isolated-bin/shum" ]] ||
  { echo "Explicit headless onboarding did not register its command" >&2; exit 1; }
run env PATH="$onboarding_command_dir:$PATH" bash -c \
  'cd "$1"; [[ $(command -v shum) == "$2" ]] && shum --help' \
  _ "$scratch" "$onboarding_command_dir/shum"
contains "Usage:" "$output"
same "$scratch/old-global-locator" "$XDG_CONFIG_HOME/shum/installation.toml"
contains "installation_dir = \"$isolated\"" "$scratch/isolated-settings/settings.toml"
run "$scratch/isolated-bin/shum" --installation "$scratch/isolated-settings/settings.toml"
contains "Shum is configured. Projects: $scratch/isolated-projects" "$output"
reject "$bin/shum" --onboarding \
  --storage-dir "$scratch/foreign-root" --bin-dir "$scratch/foreign-bin" \
  --settings-dir "$settings" --projects-dir "$scratch/foreign-projects" \
  --models-dir "$scratch/foreign-models"
contains "describe a different layout" "$output"
mkdir -- "$scratch/orphan-config"
printf 'keep existing config\n' >"$scratch/orphan-config/config.toml"
reject "$bin/shum" --onboarding \
  --storage-dir "$scratch/orphan-root" --bin-dir "$scratch/orphan-bin" \
  --settings-dir "$scratch/orphan-config" --projects-dir "$scratch/orphan-projects" \
  --models-dir "$scratch/orphan-models"
contains "without matching settings" "$output"
contains "keep existing config" "$scratch/orphan-config/config.toml"
same "$scratch/old-global-locator" "$XDG_CONFIG_HOME/shum/installation.toml"

# Reinstallation is idempotent, preserving project data and config; backup is
# unique on each retry, including a third installation of the same binary.
printf 'private project data\n' >"$projects/recording.txt"
cp -- "$settings/config.toml" "$scratch/original-config"
for attempt in 1 2; do
  run env SHUM_SETTINGS="$settings_file" "$binary" --onboarding \
    --storage-dir "$install" --bin-dir "$bin" --settings-dir "$settings" \
    --projects-dir "$projects" --models-dir "$models"
  same "$scratch/original-config" "$settings/config.toml"
  contains "private project data" "$projects/recording.txt"
done
[[ $(find "$settings" -maxdepth 1 -name 'config.toml.bak-*' | wc -l) == 2 ]] ||
  { echo "Expected separate, preserved config backups" >&2; exit 1; }

# Legacy v1 settings migrate project/model data by copying, not deleting.
old="$scratch/legacy-store"
legacy_settings="$scratch/legacy-settings.toml"
mkdir -p -- "$old/projects/take" "$old/models/primeline"
printf 'legacy transcript\n' >"$old/projects/take/result.txt"
printf 'legacy weight\n' >"$old/models/primeline/weights"
printf 'version = 1\nstorage_dir = "%s"\n' "$old" >"$legacy_settings"
legacy_new="$scratch/legacy-new"
run env SHUM_SETTINGS="$legacy_settings" "$binary" --onboarding \
  --storage-dir "$legacy_new" --bin-dir "$scratch/legacy-bin" \
  --settings-dir "$scratch/legacy-config" \
  --projects-dir "$scratch/legacy-projects" --models-dir "$scratch/legacy-models"
same "$old/projects/take/result.txt" "$scratch/legacy-projects/take/result.txt"
same "$old/models/primeline/weights" "$scratch/legacy-models/primeline/weights"
contains "legacy transcript" "$old/projects/take/result.txt"
contains "legacy weight" "$old/models/primeline/weights"
run "$scratch/legacy-bin/shum" --installation "$scratch/legacy-config/settings.toml"
contains "Shum is configured." "$output"

# Reject overlaps (including aliases) before writing outside the installation.
reject env SHUM_SETTINGS="$settings_file" "$binary" --onboarding \
  --storage-dir "$install" --bin-dir "$bin" --settings-dir "$settings" \
  --projects-dir "$projects" --models-dir "$projects/inside"
contains "overlap" "$output"
ln -s -- "$projects" "$scratch/projects-link"
reject env SHUM_SETTINGS="$settings_file" "$binary" --onboarding \
  --storage-dir "$install" --bin-dir "$bin" --settings-dir "$settings" \
  --projects-dir "$projects" --models-dir "$scratch/projects-link/inside"
contains "overlap" "$output"
same "$scratch/original-config" "$settings/config.toml"

# Missing and malformed config and runtime are recoverable errors without
# non-interactive reset or destructive overwrite.
mv -- "$settings/config.toml" "$scratch/saved-config"
reject "$bin/shum" --installation "$settings_file"
contains "Configuration repair requires a terminal" "$output"
mv -- "$scratch/saved-config" "$settings/config.toml"
printf 'invalid = [\n' >"$settings/config.toml"
reject env SHUM_SETTINGS="$settings_file" "$binary" --onboarding \
  --storage-dir "$install" --bin-dir "$bin" --settings-dir "$settings" \
  --projects-dir "$projects" --models-dir "$models"
contains "interactive repair is required" "$output"
contains "invalid = [" "$settings/config.toml"
mv -- "$scratch/original-config" "$settings/config.toml"
runtime_complete=("$install"/runtime/*/.complete)
[[ ${#runtime_complete[@]} == 1 && -f ${runtime_complete[0]} ]] ||
  { echo "Unexpected bundled runtime layout" >&2; exit 1; }
mv -- "${runtime_complete[0]}" "$scratch/runtime-complete"
reject "$bin/shum" --installation "$settings_file"
contains "runtime resources are missing or damaged" "$output"
reject env SHUM_SETTINGS="$settings_file" "$binary" --onboarding \
  --storage-dir "$install" --bin-dir "$bin" --settings-dir "$settings" \
  --projects-dir "$projects" --models-dir "$models"
contains "interactive repair is required" "$output"
[[ -f "$scratch/runtime-complete" ]] || { echo "Damaged runtime was changed" >&2; exit 1; }

echo "Onboarding smoke passed (offline, isolated; ASR models/environments not installed)."
