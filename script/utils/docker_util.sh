#!/bin/sh

case "$1" in 
    -d)
    HASH=$2
    FROM=$3
    TO=$4
    echo "Download from docker image ${HASH} from ${FROM} to ${TO}"
    docker cp ${HASH}://workspaces/band/${FROM} ${TO}
    ;;

    -r) 
    HASH=$2
    RUN_COMMAND=${@:3}
    echo "Run from band workspace in docker image ${HASH} command: ${RUN_COMMAND}"
    docker exec -i ${HASH} bash -c "cd /workspaces/band/ && ${RUN_COMMAND}"
    ;;

    *) echo "`basename $0`: usage: [-d directory] | [-r run options (using bash from workspaces/band)]"
    exit 1
    ;;
esac