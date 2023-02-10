import subprocess

def copy_remote(host, src, dst):
    subprocess.call(f'sh script/utils/ssh_util.sh -d {src} {host}:{dst}')
    
def run_cmd_remote(host, exec):
    subprocess.call(f'sh script/utils/ssh_util.sh -r {host} {exec}')