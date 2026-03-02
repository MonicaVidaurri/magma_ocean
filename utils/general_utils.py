
def merge_dicts(overwrite_dict, reference_dict):
    for key, value in reference_dict.items():
        if key not in overwrite_dict:
            # Value not in overwrite dict; use reference
            overwrite_dict[key] = value
        elif isinstance(value, dict):
            # Value is in both and it is another dict; recursive call.
            merge_dicts(overwrite_dict[key], value)
        else:
            # Value is in both; prioritize overwrite.
            pass
    return overwrite_dict