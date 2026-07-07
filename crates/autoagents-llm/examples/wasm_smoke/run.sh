#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CRATE_DIR="$ROOT_DIR/crates/autoagents-llm"
BASE_URL="http://127.0.0.1:18765/v1"

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  if [[ "$CARGO_TARGET_DIR" = /* ]]; then
    TARGET_DIR="$CARGO_TARGET_DIR"
  else
    TARGET_DIR="$ROOT_DIR/$CARGO_TARGET_DIR"
  fi
else
  TARGET_DIR="$ROOT_DIR/target"
fi

MOCK_SERVER_PATH="$TARGET_DIR/debug/examples/wasm_mock_server"
WASM_PATH="$TARGET_DIR/wasm32-wasip2/debug/examples/wasm_agent.wasm"

missing=()
command -v cargo >/dev/null 2>&1 || missing+=(cargo)
command -v wasmtime >/dev/null 2>&1 || missing+=(wasmtime)
command -v wasm-component-ld >/dev/null 2>&1 || missing+=(wasm-component-ld)
if ! rustup target list --installed 2>/dev/null | grep -qx 'wasm32-wasip2'; then
  missing+=(rust-target-wasm32-wasip2)
fi

if ((${#missing[@]})); then
  printf 'SKIP: missing prerequisites: %s\n' "${missing[*]}" >&2
  exit 2
fi

server_log="$(mktemp)"
stdout_log="$(mktemp)"
cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  rm -f "$server_log" "$stdout_log"
}
trap cleanup EXIT

(
  cd "$ROOT_DIR"
  cargo build -p autoagents-llm --features openai,wasi-http --example wasm_mock_server
  # Link as a WebAssembly *component* (not a plain module): the `wasi:http`
  # imports the agent uses are component-model interfaces, so the default
  # module linker would fail to resolve them at runtime.
  RUSTFLAGS='-C linker=wasm-component-ld' \
    cargo build -p autoagents-llm --target wasm32-wasip2 --features wasi-http,openai --example wasm_agent
)

"$MOCK_SERVER_PATH" >"$server_log" 2>&1 &
server_pid=$!

for _ in {1..300}; do
  if grep -q "Starting mock server" "$server_log"; then
    break
  fi
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    printf 'mock server exited early:\n' >&2
    cat "$server_log" >&2
    exit 1
  fi
  sleep 0.1
done
if ! grep -q "Starting mock server" "$server_log"; then
  printf 'mock server did not become ready:\n' >&2
  cat "$server_log" >&2
  exit 1
fi

(
  cd "$ROOT_DIR"
  wasmtime run -W component-model=y -S http=true \
    --env BASE_URL="$BASE_URL" \
    --env OPENAI_API_KEY=sk-test \
    --env STRICT_MOCK_EXPECTATIONS=true \
    --env RUN_ERROR_TEST=true \
    "$WASM_PATH"
) | tee "$stdout_log"

grep -q 'PROVIDER_NON_STREAM_OK' "$stdout_log"
grep -q 'PROVIDER_STREAM_OK text_delta=true reasoning_delta=true tool_call=true usage=true' "$stdout_log"
grep -q 'PROVIDER_ERROR_429_OK' "$stdout_log"
grep -q 'PROVIDER_MODELS_OK count=2 first=gpt-4.1' "$stdout_log"
grep -q 'PROVIDER_EMBED_OK count=2 dims=3' "$stdout_log"

printf 'WASM_SMOKE_OK\n'
