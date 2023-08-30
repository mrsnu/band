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

import subprocess


def get_container_hash(keyword="vsc-band"):
    if keyword in ["CONTAINER", "ID", "IMAGE", "COMMAND", "CREATED", "STATUS", "PORTS", "NAMES"]:
        return []
    return subprocess.getoutput(f"docker ps -a | grep {keyword} | awk '{{print $1}}'").split()


def copy_docker(src, dst, hash=None):
    if hash is None:
        hash = get_container_hash()[0]
    subprocess.check_call(['mkdir',  '-p', '{dst}'])
    subprocess.check_call(['sh', 'script/utils/docker_util.sh', '-d', hash, src, dst])


def run_cmd_docker(exec, hash=None):
    if hash is None:
        hash = get_container_hash()[0]
    subprocess.check_call(['sh', 'script/utils/docker_util.sh', '-r', hash, exec])
