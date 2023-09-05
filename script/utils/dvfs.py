from time import sleep
from typing import Dict

from .util import run_on_android


class DVFS(object):
    def __init__(self) -> None:
        self.cpufreq_policies = self.get_cpufreq_policies()
        self.cpufreq_available_frequencies: Dict[str, list] = {}
        for policy in self.cpufreq_policies:
            self.cpufreq_available_frequencies[policy] = self.get_cpufreq_available_freqs(policy)

    def fix_all_cpufreq_max(self):
        for policy in self.cpufreq_policies:
            max_freq = self.cpufreq_available_frequencies[policy][-1]
            self.set_cpufreq_freq(policy, max_freq)
            cpufreq = self.get_cpufreq_curfreq(policy)
            if cpufreq != max_freq:
                raise ValueError(cpufreq, max_freq)

    def fix_all_cpufreq_min(self):
        cpufreq_policies = self.get_cpufreq_policies()
        for policy in cpufreq_policies[1:]:
            min_freq = self.cpufreq_available_frequencies[policy][0]
            while True:
                self.set_cpufreq_freq(policy, min_freq)
                cpufreq = self.get_cpufreq_curfreq(policy)
                if cpufreq == min_freq:
                    break
                else:
                    self.print_cpufreq_status(policy)
                    sleep(1)

    def print_cpufreq_status(self, policy):
        governor = run_on_android(f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor', capture_output=True, run_as_su=True).strip()
        cur_freq = run_on_android(f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_cur_freq', capture_output=True, run_as_su=True).strip()
        min_freq = run_on_android(f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_min_freq', capture_output=True, run_as_su=True).strip()
        max_freq = run_on_android(f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_max_freq', capture_output=True, run_as_su=True).strip()
        available_min_freq = self.cpufreq_available_frequencies[policy][0]
        available_max_freq = self.cpufreq_available_frequencies[policy][-1]
        print(governor, policy, cur_freq, min_freq, max_freq, available_min_freq, available_max_freq)

    def get_cpufreq_policies(self):
        stdout = run_on_android(
            'ls /sys/devices/system/cpu/cpufreq',
            run_as_su=False, capture_output=True)
        return stdout.splitlines()
    
    def set_cpufreq_all_governors(self, governor: str):
        for policy in self.cpufreq_policies:
            self.set_cpufreq_governor(policy, governor)

    def set_cpufreq_governor(self, policy: str, governor: str):
        run_on_android(
            f'echo {governor} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor',
            run_as_su=True)
        if governor == 'userspace':
            available_frequencies = self.cpufreq_available_frequencies[policy]
            while True:
                self.set_cpufreq_minfreq(policy, available_frequencies[0])
                minfreq = self.get_cpufreq_minfreq(policy)
                if minfreq == available_frequencies[0]:
                    break
                else:
                    self.print_cpufreq_status(policy)
                    sleep(1)
                self.set_cpufreq_maxfreq(policy, available_frequencies[-1])
                maxfreq = self.get_cpufreq_maxfreq(policy)
                if maxfreq == available_frequencies[-1]:
                    break
                else:
                    self.print_cpufreq_status(policy)
                    sleep(1)

    def get_cpufreq_minfreq(self, policy: str):
        return int(run_on_android(
            f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_min_freq',
            capture_output=True, run_as_su=True))

    def get_cpufreq_maxfreq(self, policy: str):
        return int(run_on_android(
            f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_max_freq',
            capture_output=True, run_as_su=True))

    def set_cpufreq_minfreq(self, policy: str, min_freq: int):
        run_on_android(
            f'echo {min_freq} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_min_freq',
            run_as_su=True)

    def set_cpufreq_maxfreq(self, policy: str, min_freq: int):
        run_on_android(
            f'echo {min_freq} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_max_freq',
            run_as_su=True)

    def get_cpufreq_governor(self, policy: str):
        stdout = run_on_android(
            f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor',
            capture_output=True, run_as_su=False)
        return stdout.strip()

    def get_cpufreq_curfreq(self, policy: str):
        stdout = run_on_android(
            f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_cur_freq',
            capture_output=True)
        return int(stdout.strip())

    def get_cpufreq_available_freqs(self, policy: str):
        stdout = run_on_android(
            f'cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_available_frequencies',
            capture_output=True)
        return sorted([int(token) for token in stdout.split()])

    def set_cpufreq_freq(self, policy: str, freq: int):
        run_on_android(
            f'echo {freq} > /sys/devices/system/cpu/cpufreq/{policy}/scaling_setspeed',
            run_as_su=True)


if __name__ == '__main__':
    dvfs = DVFS()
    print(dvfs.get_cpufreq_governor(dvfs.cpufreq_policies[-1]))

    dvfs.set_cpufreq_governor(dvfs.cpufreq_policies[-1], 'userspace')

    dvfs.fix_all_cpufreq_min()
    dvfs.fix_all_cpufreq_max()

    # available_freqs = dvfs.get_cpufreq_available_freqs(dvfs.cpufreq_policies[-1])
    # dvfs.set_cpufreq_minfreq(dvfs.cpufreq_policies[-1], available_freqs[0])
    # minfreq = dvfs.get_cpufreq_minfreq(dvfs.cpufreq_policies[-1])
    # if minfreq != available_freqs[0]:
    #     raise ValueError(minfreq, available_freqs[0])

    # print(dvfs.get_cpufreq_governor(dvfs.cpufreq_policies[-1]))
    # print(dvfs.get_cpufreq_curfreq(dvfs.cpufreq_policies[-1]))
    # freqs = dvfs.get_cpufreq_available_freqs(dvfs.cpufreq_policies[-1])
    # print(freqs)
    # dvfs.set_cpufreq_freq(dvfs.cpufreq_policies[-1], freqs[-1])
    # print(dvfs.get_cpufreq_curfreq(dvfs.cpufreq_policies[-1]))

    dvfs.set_cpufreq_governor(dvfs.cpufreq_policies[-1], 'schedutil')
    print(dvfs.get_cpufreq_governor(dvfs.cpufreq_policies[-1]))
