import subprocess

def get_container_hash(keyword="vsc-band"):
    if keyword in ["CONTAINER", "ID", "IMAGE", "COMMAND", "CREATED", "STATUS", "PORTS", "NAMES"]:
        return []
    return subprocess.getoutput(f"docker ps -a | grep {keyword} | awk '{{print $1}}'").split()

def copy_docker(src, dst, hash=None):
    if hash is None:
        hash = get_container_hash()[0]
    subprocess.call(f'sh script/utils/docker_util.sh -d {hash} {src} {dst}')
    
def run_cmd_docker(exec, hash=None):
    if hash is None:
        hash = get_container_hash()[0]
    subprocess.call(f'sh script/utils/docker_util.sh -r {hash} {exec}')
    