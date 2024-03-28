#!/bin/sh
# Copyright 2023 Seoul National University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


case "$1" in 
    -d)
    HASH=$2
    FROM=$3
    TO=$4
    echo "Copy from ${HASH}:${FROM} to ${TO}"
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