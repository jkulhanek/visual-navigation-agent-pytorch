import os
import re

def find_restore_point(checkpoint_path, fail = True):
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Find latest checkpoint
    restore_point = None
    if checkpoint_path.find('{checkpoint}') != -1:
        files = os.listdir(os.path.dirname(checkpoint_path))
        base_name = os.path.basename(checkpoint_path)
        regex = re.escape(base_name).replace(re.escape('{checkpoint}'), '(\d+)')
        points = [(fname, int(match.group(1))) for (fname, match) in ((fname, re.match(regex, fname),) for fname in files) if not match is None]
        if len(points) == 0:
            if fail:
                raise Exception('Restore point not found')
            else: return None
        
        (base_name, restore_point) = max(points, key = lambda x: x[1])
        return (base_name, restore_point)
    else:
        if not os.path.exists(checkpoint_path):
            if fail:
                raise Exception('Restore point not found')
            else: return None
        return (checkpoint_path, None)