from .util import run_on_android


def fix_all_cpufreq_max():
    cpufreq_policies = get_cpufreq_policies()
    for cpufreq_policy in cpufreq_policies:
        set_cpufreq_governor(cpufreq_policy, 'userspace')
        cpu_governor = get_cpufreq_governor(cpufreq_policy)
        if cpu_governor != 'userspace':
            raise ValueError(cpu_governor)
        available_freqs = get_cpufreq_available_freqs(cpufreq_policy)
        max_freq = sorted(available_freqs)[-1]
        set_cpufreq_freq(cpufreq_policy, max_freq)
        cpufreq = get_cpufreq_curfreq(cpufreq_policy)
        if cpufreq != max_freq:
            raise ValueError(cpufreq, max_freq)


def fix_all_cpufreq_min():
    cpufreq_policies = get_cpufreq_policies()
    # Not sure why we can't set the lowest frequency for policy0 (Little Core) with this code.
    for cpufreq_policy in cpufreq_policies[1:]:
        set_cpufreq_governor(cpufreq_policy, 'userspace')
        cpu_governor = get_cpufreq_governor(cpufreq_policy)
        if cpu_governor != 'userspace':
            raise ValueError(cpu_governor)
        available_freqs = get_cpufreq_available_freqs(cpufreq_policy)
        min_freq = sorted(available_freqs)[0]
        set_cpufreq_freq(cpufreq_policy, min_freq)
        cpufreq = get_cpufreq_curfreq(cpufreq_policy)
        if cpufreq != min_freq:
            raise ValueError(cpufreq, min_freq)


def set_all_cpu_governor_schedutil():
    for policy in get_cpufreq_policies():
        set_cpufreq_governor(policy, 'schedutil')
        governor = get_cpufreq_governor(policy)
        if governor != 'schedutil':
            raise ValueError(governor)


def get_cpufreq_policies():
    stdout = run_on_android(
        'ls /sys/devices/system/cpu/cpufreq',
        run_as_su=False, capture_output=True)
    return stdout.splitlines()


def set_cpufreq_governor(policy: str, governor: str):
    run_on_android(
        f'echo {governor} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor',
        run_as_su=True)


def get_cpufreq_governor(policy: str):
    stdout = run_on_android(
        f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor',
        capture_output=True, run_as_su=False)
    return stdout.strip()


def get_cpufreq_curfreq(policy: str):
    stdout = run_on_android(
        f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_cur_freq',
        capture_output=True)
    return int(stdout.strip())


def get_cpufreq_available_freqs(policy: str):
    stdout = run_on_android(
        f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_available_frequencies',
        capture_output=True)
    return [int(token) for token in stdout.split()]


def set_cpufreq_freq(policy: str, freq: int):
    run_on_android(
        f'echo {freq} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_setspeed',
        run_as_su=True)


if __name__ == '__main__':
    policies = get_cpufreq_policies()
    print(policies)
    print(get_cpufreq_governor(policies[-1]))
    set_cpufreq_governor(policies[-1], 'userspace')
    print(get_cpufreq_governor(policies[-1]))
    print(get_cpufreq_curfreq(policies[-1]))
    freqs = get_cpufreq_available_freqs(policies[-1])
    print(freqs)
    set_cpufreq_freq(policies[-1], freqs[-1])
    print(get_cpufreq_curfreq(policies[-1]))
    set_cpufreq_governor(policies[-1], 'schedutil')
    print(get_cpufreq_governor(policies[-1]))
