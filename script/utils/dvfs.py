from .util import run_on_android


def get_cpu_governor_policies():
    stdout = run_on_android(
        'ls /sys/devices/system/cpu/cpufreq',
        run_as_su=False, capture_output=True)
    return stdout.splitlines()


def set_cpu_governor(policy: str, governor: str):
    run_on_android(
        f'echo {governor} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor',
        run_as_su=True)


def get_cpu_governor(policy: str):
    stdout = run_on_android(
        f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor',
        capture_output=True, run_as_su=False)
    return stdout.strip()


def get_cpu_freq(policy: str):
    stdout = run_on_android(
        f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_cur_freq',
        capture_output=True)
    return int(stdout.strip())


def get_cpu_available_frequencies(policy: str):
    stdout = run_on_android(
        f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_available_frequencies',
        capture_output=True)
    return [int(token) for token in stdout.split()]


def set_cpu_freq(policy: str, freq: int):
    run_on_android(
        f'echo {freq} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_setspeed',
        run_as_su=True)


if __name__ == '__main__':
    policies = get_cpu_governor_policies()
    print(policies)
    print(get_cpu_governor(policies[-1]))
    set_cpu_governor(policies[-1], 'userspace')
    print(get_cpu_governor(policies[-1]))
    print(get_cpu_freq(policies[-1]))
    freqs = get_cpu_available_frequencies(policies[-1])
    print(freqs)
    set_cpu_freq(policies[-1], freqs[-1])
    print(get_cpu_freq(policies[-1]))
    set_cpu_governor(policies[-1], 'schedutil')
    print(get_cpu_governor(policies[-1]))
