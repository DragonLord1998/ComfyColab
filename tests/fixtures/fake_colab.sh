#!/bin/sh
set -eu

command_name=""
for argument in "$@"; do
  case "$argument" in
    status|new|exec|stop)
      command_name="$argument"
      break
      ;;
  esac
done

case "$command_name" in
  status)
    echo "[colab] Session 'comfycolab' not found."
    ;;
  new)
    echo "[colab] Session READY."
    ;;
  exec)
    echo 'COMFYCOLAB_READY={"status":"ready","comfyUrl":"https://fake.trycloudflare.com"}'
    ;;
  stop)
    echo "[colab] Session terminated."
    ;;
  *)
    echo "Unexpected fake colab command: $*" >&2
    exit 2
    ;;
esac
