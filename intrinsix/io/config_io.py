from omegaconf import ListConfig


def range2list(config, max_length=None):
    if config is None:
        return list(range(max_length))
    elif isinstance(config, int):
        return [config]
    elif isinstance(config, list):
        return config
    elif isinstance(config, ListConfig):
        return config
    elif isinstance(config, str):
        slice_def = config.split(":")

        start = 0
        if len(slice_def) >= 1:
            if slice_def[0] != "":
                start = int(slice_def[0])

        end = max_length
        if len(slice_def) >= 2:
            if slice_def[1] != "":
                end = int(slice_def[1])
        assert end is not None, f"End must be defined for config {config}"

        step = 1
        if len(slice_def) == 3:
            if slice_def[2] != "":
                step = int(slice_def[2])

        return list(range(start, end, step))
