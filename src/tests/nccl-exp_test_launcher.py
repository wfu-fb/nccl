import argparse
import contextlib
import logging
import os
import socket
import subprocess
import sys

from pathlib import Path
from subprocess import CompletedProcess
from typing import Dict, Iterable, Tuple


def exec_cmds(run_cmds: Iterable[str]) -> "list[CompletedProcess[bytes]]":
    """execute command lines on shell"""
    shell_outputs = []
    for cmd in run_cmds:
        logging.info(f"Running {cmd}")
        shell_outputs.append(
            subprocess.run(
                [cmd],
                shell=True,
                check=True,
                stderr=subprocess.STDOUT,
            )
        )
    return shell_outputs


def get_nccl_test_binary(name: str) -> Tuple[Path, str]:
    """get full path and executable name from par package"""
    # If it looks like a path, treat it like one
    # This is useful in non-opt modes where FB_PAR_RUNTIME_FILES
    # isn't available
    if "/" in name:
        p = Path(name)
        return p.parent, p.name

    par_path = os.getenv("FB_PAR_RUNTIME_FILES")
    if par_path is None:
        par_path = Path.cwd()
    else:
        par_path = Path(par_path)

    return par_path, name

def parse_envs(env_str: str) -> Dict[str, str]:
    """parse environment variables and convert it into a dictionary mapping"""
    envs_dict = {}
    if env_str is not None:
        for e in env_str.split(";"):
            # skip invalid env var or if reached the ending determiner
            if e and "=" in e:
                key = e.split("=")[0]
                val = e.split("=")[1]
                envs_dict[key] = val
    return envs_dict


def find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def remote_mpi_launcher(args: argparse.Namespace) -> None:
    """launch multi-process/node runs on remote hosts using MPI launcher"""
    if len(args.hosts.split(",")) != args.nnode:
        logging.warning(
            f"nnode ({args.nnode}) does not match provided hosts: {args.hosts}...use provided hosts"
        )
        args.nnode = len(args.hosts.split(","))

    master_addr = os.environ.get("MASTER_ADDR", args.hosts.split(",")[0])
    if args.mpi_args is None:
        np = args.nnode * args.ppn
        ifname = args.ifname
        host_list = [f"{host}:{args.ppn}" for host in args.hosts.split(",")]
        final_hosts = ",".join(host_list)
        mpi_args = f"-np {np} -host {final_hosts} --allow-run-as-root -x MASTER_ADDR={master_addr}"
        if master_addr not in ("localhost", "127.0.0.1") and args.nnode > 1:
            mpi_args = f"{mpi_args} -x THRIFT_TLS_CL_KEY_PATH=/var/facebook/x509_identities/server.pem -x THRIFT_TLS_CL_CERT_PATH=/var/facebook/x509_identities/server.pem --gmca btl_tcp_if_include {ifname} --gmca oob_tcp_if_include {ifname} --gmca btl tcp,self"
    else:
        mpi_args = args.mpi_args

    # env. variables to be set on remote hosts
    envs = parse_envs(args.envs)
    # If user doesn't specify MASTER_PORT, try to dynamically find a free one on local machine where launcher is called
    # Note that this allows two nccl-tests-launcher to concurrently run on the same machine without port conflict (e.g., on sandcastle)
    if "MASTER_PORT" not in envs.keys():
        envs["MASTER_PORT"] = f"{find_free_port()}"
        logging.info(f"Pick free port {envs['MASTER_PORT']}")

    for key, val in envs.items():
        mpi_args += f" -x {key}={val}"

    run_cmds = []
    for coll in args.testname.split(","):
        par_path, executable = get_nccl_test_binary(coll)
        logging.info(f"Launching nccl-exp-test at {par_path}/{executable}")
        # copy binary to remote hosts
        for host in args.hosts.split(","):
            if host != "localhost":
                run_cmds.append(
                    f"suscp --reason 'copy NCCL-EXP test launcher binary' {par_path}/{executable} root@{host}:/tmp/"
                )
        # Default run
        # mpirun launching nccl-tests on remote node
        if master_addr != "localhost":
            run_cmds.append(
                f"sush2 --reason 'Testing NCCL-EXP test' root@{master_addr} '/usr/local/fbcode/bin/mpirun {mpi_args} /tmp/{executable}'"
            )
        else:
            run_cmds.append(
                f"/usr/local/fbcode/bin/mpirun {mpi_args} {par_path}/{executable}"
            )

        # additional runs if sweep config is given
        # mpirun launching nccl-tests on remote node
        if master_addr != "localhost":
            run_cmds.append(
                f"sush2 --reason 'tesing NCCL-EXP test' root@{master_addr} '/usr/local/fbcode/bin/mpirun {mpi_args} /tmp/{executable}'"
            )
        else:
            run_cmds.append(
                f"/usr/local/fbcode/bin/mpirun {mpi_args} {par_path}/{executable}"
            )

    exec_cmds(run_cmds)


def init_argparse() -> argparse.ArgumentParser:
    """parsing arguments"""
    parser = argparse.ArgumentParser(
        usage="%(prog)s -- [OPTIONS]",
        description="NCCL-EXP test launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--testname",
        "--test",
        type=str,
        default="",
        help=f"name of the test target",
    )
    parser.add_argument(
        "--nnode",
        "-nnode",
        type=int,
        default=1,
        help="number of nodes",
    )
    parser.add_argument(
        "--ppn",
        "-N",
        type=int,
        default=8,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="number of processes/GPUs per node",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default="localhost",
        help="hosts for MPI to launch nccl-tets, separated by comma",
    )
    parser.add_argument(
        "--mpi-args",
        type=str,
        default=None,
        help="arguments to be passed to MPI launcher if used (wrapped by single/double quotes)",
    )
    parser.add_argument(
        "--envs",
        "--env",
        type=str,
        default=None,
        help="environment variables pass to MPI or MAST runs, single string separated by semicolon (;) and wrapped by single/double quotes",
    )
    parser.add_argument(
        "--ifname",
        "--tcp-if",
        type=str,
        default="eth2",
        help="Front-end interface for MPI launcher",
    )
    return parser


def main() -> None:
    global args
    # set log level
    numeric_level: int = getattr(logging, "INFO", None)
    logging.basicConfig(level=numeric_level)
    logging.info(f"Launching nccl-tests with {sys.argv}")

    parser: argparse.ArgumentParser = init_argparse()
    args = parser.parse_args(sys.argv[1:])

    # get binary inside the par file and use MPI launcher
    remote_mpi_launcher(args)


args: argparse.Namespace


if __name__ == "__main__":
    main()
