from utils.util import run_on_android


def get_thermal_zone_names():
    out = run_on_android('ls /sys/class/thermal',
                         run_as_su=True, capture_output=True)
    tzs = filter(lambda s: s.startswith('thermal_zone'), out.splitlines())
    tz_name_dict = {
        tz: run_on_android(
            f'cat /sys/class/thermal/{tz}/type', run_as_su=True,
            capture_output=True
            ).strip()
        for tz in tzs}
    return tz_name_dict


if __name__ == '__main__':
    import json

    with open('benchmark/thermal_zone_names.json', 'w') as fp:
        json.dump(get_thermal_zone_names(), fp)
