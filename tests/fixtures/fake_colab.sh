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
    if [ "${EXPECT_REFRESH:-0}" = "1" ]; then
      encoded="$(sed -n 's/^CONFIG_B64 = "\([A-Za-z0-9+/=]*\)"$/\1/p' "$bootstrap_file")"
      decoded="$(printf '%s' "$encoded" | base64 --decode 2>/dev/null || printf '%s' "$encoded" | base64 -D)"
      printf '%s' "$decoded" | grep -Fq '"refresh":true'
    fi
    if [ "${EXPECT_COLAB_PROXY:-0}" = "1" ]; then
      encoded="$(sed -n 's/^CONFIG_B64 = "\([A-Za-z0-9+/=]*\)"$/\1/p' "$bootstrap_file")"
      decoded="$(printf '%s' "$encoded" | base64 --decode 2>/dev/null || printf '%s' "$encoded" | base64 -D)"
      printf '%s' "$decoded" | grep -Fq '"colab_proxy":true'
      echo 'COMFYCOLAB_READY={"status":"ready","comfyUrl":"https://fake-8188.colab.googleusercontent.com/","cloudflareUrl":"https://fake.trycloudflare.com","colabProxyUrl":"https://fake-8188.colab.googleusercontent.com/"}'
    else
      echo 'COMFYCOLAB_READY={"status":"ready","comfyUrl":"https://fake.trycloudflare.com","cloudflareUrl":"https://fake.trycloudflare.com","colabProxyUrl":null}'
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
