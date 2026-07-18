#!/bin/sh
set -eu

command_name=""
bootstrap_file=""
previous=""
for argument in "$@"; do
  if [ "$previous" = "--file" ]; then
    bootstrap_file="$argument"
  fi
  case "$argument" in
    status|new|exec|stop|url)
      command_name="$argument"
      ;;
  esac
  previous="$argument"
done

case "$command_name" in
  status)
    echo "[colab] Session 'comfycolab' not found."
    ;;
  new)
    echo "[colab] Session READY."
    ;;
  url)
    echo "https://colab.research.google.com/notebooks/empty.ipynb?dbu=fake"
    ;;
  exec)
    encoded="$(sed -n 's/^CONFIG_B64 = "\([A-Za-z0-9+/=]*\)"$/\1/p' "$bootstrap_file")"
    decoded="$(printf '%s' "$encoded" | base64 --decode 2>/dev/null || printf '%s' "$encoded" | base64 -D)"
    lock_sha256="$(printf '%s' "$decoded" | sed -n 's/.*"lock_sha256":"\([0-9a-f]*\)".*/\1/p')"
    if [ -z "$lock_sha256" ]; then
      echo "Missing stage-0 lock digest." >&2
      exit 1
    fi
    if [ "${EXPECT_REFRESH:-0}" = "1" ]; then
      printf '%s' "$decoded" | grep -Fq '"refresh":true'
    fi
    if [ "${EXPECT_COLAB_PROXY:-0}" = "1" ]; then
      printf '%s' "$decoded" | grep -Fq '"colab_proxy":true'
      printf 'COMFYCOLAB_READY={"status":"ready","comfyUrl":"https://fake-8188.colab.googleusercontent.com/","cloudflareUrl":"https://fake.trycloudflare.com","colabProxyUrl":"https://fake-8188.colab.googleusercontent.com/","lockSha256":"%s"}\n' "$lock_sha256"
    else
      printf 'COMFYCOLAB_READY={"status":"ready","comfyUrl":"https://fake.trycloudflare.com","cloudflareUrl":"https://fake.trycloudflare.com","colabProxyUrl":null,"lockSha256":"%s"}\n' "$lock_sha256"
    fi
    ;;
  stop)
    echo "[colab] Session terminated."
    ;;
  *)
    echo "Unexpected fake colab command: $*" >&2
    exit 2
    ;;
esac
