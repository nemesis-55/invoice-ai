#!/bin/bash
PREV_CHECKPOINT=${1:-""}
EXTRA_ARGS=""
if [ -n "$PREV_CHECKPOINT" ]; then
    echo "Resuming incremental training from: $PREV_CHECKPOINT"
    EXTRA_ARGS="--adapter_name_or_path $PREV_CHECKPOINT"
fi
llamafactory-cli train llamafactory_config.yaml $EXTRA_ARGS
