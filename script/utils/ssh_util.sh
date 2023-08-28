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