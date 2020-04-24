import os
import argparse
import re


def find_files(dir_path, suff):
    """Search a directory and its children, recursively, for files matching any suffix in `suff`

    Args:
        dir_path (str): Top-level folder to start searching from
        suff (list of str): Lowercase suffixes (usually filename extensions) to search for

    Returns:
        list of str: List of filenames with relative paths for files found
    """
    if not os.path.isdir(dir_path):
        return [dir_path] if any(dir_path.lower().endswith(s) for s in suff) else []  # It's a file
    print('Searching', dir_path)
    return [f for p in os.listdir(dir_path) for f in find_files(os.path.join(dir_path, p), suff)]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Make files for FFmpeg to use for concatenating original video files')
    argparser.add_argument('in_path', type=str, help='Folder/filename path for input')
    argparser.add_argument('out_path', type=str, help='Folder where output files will be written')
    args = argparser.parse_args()

    fnames = find_files(args.in_path, ['.mov'])
    fnames = [f for f in fnames if ('Green' in f or 'Red' in f) and 'Original' in f]
    cwd = os.getcwd()
    fnames = [os.path.join(cwd, f) for f in fnames]  # Prepend full pathname

    # Figure out which videos belong to the same class session
    session_fname_map = {}
    for fname in fnames:
        prefix = fname[:fname.rindex('/')]
        if prefix not in session_fname_map:
            session_fname_map[prefix] = []
        session_fname_map[prefix].append(fname)

    # Write one FFmpeg concat file per class session
    ffmpeg_commands = []
    for session, session_fnames in session_fname_map.items():
        out_fname = re.sub(r'\W+', '_', session)
        with open(os.path.join(args.out_path, out_fname + '.txt'), 'w') as outfile:
            outfile.write('\n'.join(['file "' + f + '"' for f in session_fnames]))
        ffmpeg_commands.append('ffmpeg -safe 0 -f concat -i ' + out_fname + '.txt -c copy ' +
                               out_fname + '.mov')
    with open(os.path.join(args.out_path, 'run_ffmpeg.bat'), 'w') as outfile:
        outfile.write('\n'.join(ffmpeg_commands) + '\n')
