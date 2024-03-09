#!/bin/sh

case "$1" in
    -d)
    FROM=$2
    TO=$3
    echo "Copying ${FROM} to ${TO}"
    scp -r ${FROM} ${TO}
    ;;
    -r)
    HOST=$2
    RUN_COMMANND=${@:3}
    echo "Run command: ${RUN_COMMAND}"
    ssh ${HOST} ${RUN_COMMAND}
    ;;
    *) echo "`basename $0`: usage: [-d directory] | [-r run options (using bash from workspaces/band)]"
    exit 1
    ;;
esac