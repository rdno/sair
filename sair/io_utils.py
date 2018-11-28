#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import errno
import glob
import os
import shutil
import subprocess
from itertools import chain


class Path(str):
    def __new__(cls, value, folder=None):
        if value is not None:
            value = expand(value)
            if folder is not None and not value.startswith("/"):
                value = Path(folder) / value
            return super(Path, cls).__new__(cls, value)
        else:
            return None

    def __truediv__(self, other):
        if isinstance(other, Path) or \
           isinstance(other, str) or isinstance(other, unicode):
            return self.join(other)
        raise Exception("Unsupported Operation.")

    def __div__(self, other):
        return self.__truediv__(other)

    def join(self, *others):
        if len(others) == 0:
            return self
        else:
            return Path(os.path.join(self, others[0])).join(*others[1:])
        return Path


def expand(path):
    return os.path.expanduser(path)


def makedir(folder):
    """Make directory and do not throw error if it exists."""
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_files(pattern):
    return glob.glob(pattern)


def is_a_file(filename):
    return os.path.isfile(filename)


def is_a_folder(folder):
    return os.path.isdir(folder)


def remove_folder(folder):
    """Remove folder and its contents."""
    shutil.rmtree(folder, ignore_errors=True)


def remove_file(filename):
    """Remove file"""
    os.remove(filename)


def clean_mkdir(folder):
    """Remove directory if it exists."""
    remove_folder(folder)
    makedir(folder)


def run_command(command, args=[], np=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
                on_error=None):
    """Run a command and wait for oit to finish.

    If `np` > 1, it runs the command using mpirun.
    """
    if isinstance(stdout, Path):
        outfile = stdout
        stdout = open(outfile, "w")
    if isinstance(stderr, Path):
        errfile = stderr
        stderr = open(errfile, "w")

    args = [str(arg) for arg in args]

    if np <= 1:
        p = subprocess.Popen(chain([command], args),
                             stdout=stdout,
                             stderr=stderr,
                             cwd=cwd)
    else:
        p = subprocess.Popen(chain(["mpirun", "-np", str(np), command], args),
                             stdout=stdout,
                             stderr=stderr,
                             cwd=cwd)

    p.wait()

    if isinstance(stdout, file):
        stdout.close()
    if isinstance(stderr, file):
        stderr.close()

    if on_error is not None:
        with open(errfile) as f:
            err = f.read().rstrip()

        if err != "":
            print(err)
            raise Exception(on_error)

    return p


def basename(filename):
    """Get basename"""
    return os.path.basename(filename)


def copy_folder(src_folder, dst_folder, clean=False):
    """Copy a folder."""
    if clean:
        remove_folder(dst_folder)
    shutil.copytree(src_folder, dst_folder)


def copy_file(src_filename, dst_filename):
    """Copy a file"""
    shutil.copy(src_filename, dst_filename)


def copy_files(files, dst_folder):
    """Copy `files` (wildcard) to `dst_folder`"""
    for f in glob.glob(files):
        shutil.copy(f, dst_folder)


def move_files(files, dst_folder):
    """Move `files` (wildcard) to `dst_folder`"""
    for f in get_files(files):
        shutil.move(f, dst_folder)


def remove_files(files):
    """Remove `files` (wildcard)"""
    for f in get_files(files):
        os.remove(f)


def dirname_of(filename):
    return Path(os.path.dirname(os.path.abspath(filename)))


def makedir_for(filename):
    """Creates the parent directory for the file"""
    makedir(dirname_of(filename))
