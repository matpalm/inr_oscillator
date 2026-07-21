from pathlib import Path
from collections import namedtuple
from functools import cache


def zarr_base_path_for(run: str, check_exists: bool = True):
    p = Path(__file__).parent.parent / "parametric_capture" / "runs" / run
    if check_exists and (not p.exists()):
        raise Exception(f"zarr_path_for path [{p}] doesn't exist?")
    return p


def model_data_z_path_for(run: str, check_exists: bool = True):
    p = zarr_base_path_for(run, check_exists) / "model_data.z"
    if check_exists and (not p.exists()):
        raise Exception(f"model_data.z path [{p}] doesn't exist?")
    return str(p)


@cache
def zarr_buffer_fields(zarr_name):

    def _fields(fields):
        return namedtuple("Fields", fields)(*range(len(fields)))

    match zarr_name:
        case "cv_buffers.z":
            return _fields(["a_cv", "b_cv", "morph_cv", "v_oct"])
        case "capture_buffers.z":
            return _fields(["morph_out", "a_out", "b_out", "tri_out"])
        case "model_data.z":
            return _fields(
                [
                    "x_tri",
                    "x_sin_q",
                    "x_cos_q",
                    "x_a_cv",
                    "x_b_cv",
                    "x_morph_cv",
                    "y_true",
                ]
            )
        case "inr_model_data.z":
            return _fields(
                [
                    "x_phase",
                    "x_a_cv",
                    "x_b_cv",
                    "x_morph_cv",
                    "y_true",
                ]
            )
        case "model_data_t.z":
            raise Exception("don't use _t anymore")
        case _:
            raise Exception("TODO: support zarr_name")


# FieldNamesType = namedtuple('FieldName', foo.keys())

#         ("cv_buffers.z", ["a_cv", "b_cv", "morph", "v/oct"]),
#         ("capture_buffers.z", ["morph out", "a out", "b out", "tri out"]),
#         ("model_data.z", ["x_tri", "x_a_cv", "x_b_cv", "x_morph", "y_true"]),
#         (
#             "model_data_t.z",
#             ["x_tri", "x_a_cv", "x_b_cv", "x_morph", "y_true", "y_pred_teacher"],
#         ),
