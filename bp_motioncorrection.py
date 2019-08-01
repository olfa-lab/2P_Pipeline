"""Create and submit SLURM job script to motion correct set of TIFFs.
"""
import argparse
import logging
import os
import sys

from SlurmScript import SlurmScript

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)

MATLAB_SCRIPT_NAME = "normcorremotioncorrection_single"
MC_PAYLOAD = """module purge
module load matlab/R2018a

cd {workingDir}

tifs=(`pwd`/{pattern})

tif=${{tifs[$SLURM_ARRAY_TASK_ID]}}

{{
    echo $tif
    matlab -nodisplay -r "{matlab_script}('$tif','{refName}'); exit"
}} > $tif.log 2>&1
"""


def make_mc_submit_script(
    params,
    fname="submit_motioncorrection.sh",
    payload=MC_PAYLOAD,
    matlab_script=MATLAB_SCRIPT_NAME,
):
    """Create TIFF motion correction submit script for SLURM

    Arguments:
        params {dict} -- Dict of parameters for the script template. 'extendedglob' and
            'numTIFFsMinusOne' added automatically.

    Keyword Arguments:
        fname {str} -- Filename for created script. (default:
            {"submit_motioncorrection.sh"})
        template {str} -- Template script. (default: {TEMPLATE_SCRIPT})

    Raises:
        RuntimeError: Raised if extended globbing feature ! is included in pattern.

    Returns:
        str -- Filepath of created script.
    """
    # get number of TIFs
    scommand = f"ls {os.path.join(params['workingDir'], '')}{params['pattern']} | wc -l"
    ls_result = SlurmScript.run_shell_command(scommand, capture_output=True)
    params["array"] = range(int(ls_result.stdout))

    params["matlab_script"] = matlab_script
    fname = os.path.abspath(os.path.expanduser(fname))
    script = SlurmScript(fname, payload, "MotionCorrection", **params)
    script.save()
    return script


def get_mc_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--workingDir", default=".", help="Working directory containing tifs to correct"
    )
    parser.add_argument(
        "--pattern",
        # Default will include all TIFF
        default="*.tif",
        help="File name pattern to select tifs, i.e. Run0034_00*.tif.",
    )
    parser.add_argument(
        "--refName",
        default="Ref.tif",
        help="File name of reference tif to be used by motion correction script.",
    )
    try:
        parser.add_argument(
            "--email",
            default=f'{os.getenv("USER")}@nyu.edu',
            help="Notification email address.",
        )
    except OSError:
        parser.add_argument(
            "--email",
            required=True,
            help="Notification email address. Entry required on your system.",
        )
    parser.add_argument(
        "--run", action="store_true", help="Run the submit script after creating."
    )
    return parser


def main(args):
    script = make_mc_submit_script(args)
    if args["run"]:
        result = script.run()
        return result


if __name__ == "__main__":
    parser = get_mc_parser()
    args = parser.parse_args()
    main(vars(args))
