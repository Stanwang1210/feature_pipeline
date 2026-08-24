"""
Verification helpers for checking whether a video's feature extraction
output already exists and is actually complete (not truncated/corrupted),
so the pipeline can resume from where it left off instead of starting over.
"""

import pyarrow.ipc as ipc

from file_writer_arrow import getFilePath


def is_output_valid(video_path, root_dir, feature_type):
    """
    Return True if the Arrow output file for (video_path, feature_type)
    exists and can be opened as a well-formed, non-empty Arrow IPC file.

    An interrupted/failed extraction never produces a file with a valid
    Arrow footer, so this naturally rejects truncated files left behind by
    a crash, in addition to files that were never written at all.
    """
    try:
        out_path = getFilePath(video_path, root_dir, feature_type)
    except ValueError:
        return False

    if not out_path.exists() or out_path.stat().st_size == 0:
        return False

    try:
        with open(out_path, "rb") as f:
            reader = ipc.open_file(f)
            return reader.num_record_batches > 0
    except Exception:
        return False


def is_video_complete(video_path, root_dir, feature_types):
    """True if every requested feature type has a valid output for this video."""
    return all(is_output_valid(video_path, root_dir, ft) for ft in feature_types)


def filter_incomplete_videos(video_paths, root_dir, feature_types):
    """
    Split video_paths into (already_done, still_needed) based on verified
    Arrow output, so a re-run only processes what's missing or corrupted.
    """
    done, todo = [], []
    for video_path in video_paths:
        if is_video_complete(video_path, root_dir, feature_types):
            done.append(video_path)
        else:
            todo.append(video_path)
    return done, todo
