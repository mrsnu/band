#!/bin/sh

CONTAINER_NAME=$(docker ps | grep vsc-band | awk 'NF>1{print $NF}')

case "$1" in 
    -d)
    FROM=$2
    TO=$3
    echo "Download from docker image ${CONTAINER_NAME} from ${FROM} to ${TO}"
    docker cp $CONTAINER_NAME://workspaces/band/${FROM} ${TO}
    ;;

    -r) 
    RUN_COMMAND=${@:2}
    echo "Run from band workspace in docker image ${CONTAINER_NAME} command: ${RUN_COMMAND}"
    docker exec -i ${CONTAINER_NAME} bash -c "cd workspaces/band/ && ${RUN_COMMAND}"
    ;;

    *) echo "`basename $0`: usage: [-d directory] | [-r run options (using bash from workspaces/band)]"
    exit 1
    ;;
esac