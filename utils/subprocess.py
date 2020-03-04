import os
import yaml
import subprocess
import numpy as np
from io import IOBase
from six.moves import shlex_quote
from six.moves import cPickle as pickle

from utils.misc import logging_rank


def process_in_parallel(tag, total_range_size, binary, cfg, ckpt_path):
    """Run the specified binary NUM_GPUS times in parallel, each time as a
    subprocess that uses one GPU. The binary must accept the command line
    arguments `--range {start} {end}` that specify a data processing range.
    """
    # subprocesses
    cfg_file = os.path.join(ckpt_path, 'test', '{}_range_config.yaml'.format(tag))
    with open(cfg_file, 'w') as f:
        yaml.dump(cfg, stream=f)
    subprocess_env = os.environ.copy()
    processes = []
    # Determine GPUs to use
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        gpu_inds = list(map(int, cuda_visible_devices.split(',')))
        assert -1 not in gpu_inds, \
            'Hiding GPU indices using the \'-1\' index is not supported'
    else:
        raise NotImplementedError
    subinds = np.array_split(range(total_range_size), len(gpu_inds))
    # Run the binary in cfg.NUM_GPUS subprocesses
    for i, gpu_ind in enumerate(gpu_inds):
        start = subinds[i][0]
        end = subinds[i][-1] + 1
        subprocess_env['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
        cmd = ('python {binary} --range {start} {end} --cfg {cfg_file} --gpu_id {gpu_id}')
        cmd = cmd.format(
            binary=shlex_quote(binary),
            start=int(start),
            end=int(end),
            cfg_file=shlex_quote(cfg_file),
            gpu_id=str(gpu_ind),
        )
        logging_rank('{} range command {}: {}'.format(tag, i, cmd))
        if i == 0:
            subprocess_stdout = subprocess.PIPE
        else:
            filename = os.path.join(ckpt_path, 'test', '%s_range_%s_%s.stdout' % (tag, start, end))
            subprocess_stdout = open(filename, 'w')
        p = subprocess.Popen(
            cmd,
            shell=True,
            env=subprocess_env,
            stdout=subprocess_stdout,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
        processes.append((i, p, start, end, subprocess_stdout))
    # Log output from inference processes and collate their results
    outputs = []
    for i, p, start, end, subprocess_stdout in processes:
        log_subprocess_output(i, p, ckpt_path, tag, start, end)
        if isinstance(subprocess_stdout, IOBase):
            subprocess_stdout.close()
        range_file = os.path.join(ckpt_path, 'test', '%s_range_%s_%s.pkl' % (tag, start, end))
        range_data = pickle.load(open(range_file, 'rb'))
        outputs.append(range_data)
    return outputs


def log_subprocess_output(i, p, ckpt_path, tag, start, end):
    """Capture the output of each subprocess and log it in the parent process.
    The first subprocess's output is logged in realtime. The output from the
    other subprocesses is buffered and then printed all at once (in order) when
    subprocesses finish.
    """
    outfile = os.path.join(ckpt_path, 'test', '%s_range_%s_%s.stdout' % (tag, start, end))
    logging_rank('# ' + '-' * 76 + ' #')
    logging_rank('stdout of subprocess %s with range [%s, %s]' % (i, start + 1, end))
    logging_rank('# ' + '-' * 76 + ' #')
    if i == 0:
        # Stream the piped stdout from the first subprocess in realtime
        with open(outfile, 'w') as f:
            for line in iter(p.stdout.readline, b''):
                print(line.rstrip().decode('ascii'))
                f.write(str(line, encoding='ascii'))
        p.stdout.close()
        ret = p.wait()
    else:
        # For subprocesses >= 1, wait and dump their log file
        ret = p.wait()
        with open(outfile, 'r') as f:
            print(''.join(f.readlines()))
    assert ret == 0, 'Range subprocess failed (exit code: {})'.format(ret)
