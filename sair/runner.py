#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import argparse
import re

from functools import partial
from . import io_utils as io
from . import config
from . import linesearch
from . import adjoint
from . import model_misfit
from . import utils

from itertools import chain


def get_nproc(model_folder):
    with open(model_folder / "Par_file") as f:
        for line in f:
            if line.lower().startswith("nproc"):
                return int(line.split("#")[0].split("=")[1])

    raise Exception("NPROC couldn't be found")


class Specfem2D(object):
    """Documentation for Specfem2D

    """

    def __init__(self, conf, model):
        super(Specfem2D, self).__init__()
        self.specfem_folder = conf.specfem_folder
        self.specfem_bin = self.specfem_folder / "bin"
        self.work_dir = io.Path(conf.root_work_dir)
        self.model_folder = io.Path(model)
        self.sources = conf.get_event_names()
        self.sta_folder = io.Path(conf.station_folder)
        self.src_folder = io.Path(conf.source_folder)
        self.nproc = get_nproc(self.model_folder)
        self.n_parallel_src_runs = conf.options.n_parallel_src_runs
        self._specfem_vars = {
            "DT": conf.simulation.dt,
            "NSTEP": conf.simulation.nstep
        }
        if conf.simulation.seismogram_format == "ascii":
            self.set_specfem_vars({
                "save_ASCII_seismograms": ".true.",
                "save_binary_seismograms_single": ".false.",
                "save_binary_seismograms_double": ".false.",
                "SU_FORMAT": ".false."
            })
        elif conf.simulation.seismogram_format == "su":
            self.set_specfem_vars({
                "save_ASCII_seismograms": ".false.",
                "save_binary_seismograms_single": ".true.",
                "save_binary_seismograms_double": ".false.",
                "SU_FORMAT": ".true."
            })
        else:
            raise Exception("Unknown seismogram format: {}".format(
                conf.simulation.seismogram_format))
        self.kernels_file = self.work_dir / "kernel_folders"

    def set_specfem_vars(self, parameters):
        self._specfem_vars.update(parameters)

    def _apply_variables(self, data_dir):
        with open(data_dir / "Par_file") as f:
            pars = f.read()

        for varname, value in self._specfem_vars.iteritems():
            pat = re.compile(
                "(^{varname}\s*=\s*)([^#$\s]+)".format(varname=varname),
                re.MULTILINE)
            pars = pat.sub("\g<1>{value}".format(value=value), pars)

        with open(data_dir / "Par_file", "w") as f:
            f.write(pars)

        self.nproc = get_nproc(data_dir)

    def _prepare(self, srcname):
        cwd = self.work_dir / "events" / srcname
        data_dir = cwd / "DATA"
        logs_dir = cwd / "logs"

        io.makedir(logs_dir)
        io.copy_folder(self.model_folder, data_dir, clean=True)

        io.copy_file(self.sta_folder / "STATIONS_{}".format(srcname),
                     data_dir / "STATIONS")
        io.copy_file(self.src_folder / "SOURCE_{}".format(srcname),
                     data_dir / "SOURCE")

        self._apply_variables(data_dir)

        return cwd, data_dir, logs_dir

    def run_mesher(self, work_dir, logs_dir):
        out_file = logs_dir / "mesher_output"
        err_file = logs_dir / "mesher_error"
        io.makedir(work_dir / "OUTPUT_FILES")

        io.run_command(self.specfem_bin / "xmeshfem2D",
                       cwd=work_dir,
                       stdout=out_file,
                       stderr=err_file,
                       on_error="Mesher Error!")

    def run_specfem(self, work_dir, logs_dir):
        out_file = logs_dir / "specfem_output"
        err_file = logs_dir / "specfem_error"

        io.run_command(self.specfem_bin / "xspecfem2D",
                       np=self.nproc,
                       cwd=work_dir,
                       stdout=out_file, stderr=err_file,
                       on_error="Specfem2D Error!")

    def run_forward_sim(self, source, output_folder, keep_lastframe=False):
        self._specfem_vars["SIMULATION_TYPE"] = 1
        self._specfem_vars["SAVE_FORWARD"] = ".false."
        if keep_lastframe:
            self._specfem_vars["SAVE_FORWARD"] = ".true."

        work_dir, data_dir, logs_dir = self._prepare(source)

        self.run_mesher(work_dir, logs_dir)
        self.run_specfem(work_dir, logs_dir)

        # Move seismograms to output folder
        target_folder = work_dir / output_folder
        io.clean_mkdir(target_folder)
        io.move_files(work_dir / "OUTPUT_FILES" / "*.semd",
                      target_folder)
        io.move_files(work_dir / "OUTPUT_FILES" / "*.su",
                      target_folder)

        if keep_lastframe:
            io.clean_mkdir(work_dir / "SEM")

    def run_backward_sim(self, source, output_folder):
        self._specfem_vars["SIMULATION_TYPE"] = 3
        self._specfem_vars["SAVE_FORWARD"] = ".false."

        work_dir, data_dir, logs_dir = self._prepare(source)
        self.run_mesher(work_dir, logs_dir)
        self.run_specfem(work_dir, logs_dir)

        # Move kernels to output folder
        target_folder = work_dir / output_folder
        io.clean_mkdir(target_folder)
        io.copy_files(data_dir / "proc*.bin",
                      target_folder)
        io.move_files(work_dir / "OUTPUT_FILES" / "proc*_kernel.bin",
                      target_folder)

    def run_all_forward(self, output_folder, keep_lastframe=False):
        p = utils.get_pool(self.n_parallel_src_runs)
        func = partial(run_forward_sim, specfem=self,
                       output_folder=output_folder,
                       keep_lastframe=keep_lastframe)
        for _ in utils.tqdm(p.imap_unordered(func, self.sources),
                            total=len(self.sources),
                            desc="Forward Simulations"):
            pass
        p.close()

    def run_all_backward(self, output_folder):
        p = utils.get_pool(self.n_parallel_src_runs)
        func = partial(run_backward_sim, specfem=self,
                       output_folder=output_folder)
        for _ in utils.tqdm(p.imap_unordered(func, self.sources),
                            total=len(self.sources),
                            desc="Backward Simulations"):
            pass

        # Write kernel folders to a file
        kernels_file = self.work_dir / "kernel_folders"
        with open(kernels_file, "w") as f:
            for source in self.sources:
                f.write("{}\n".format(
                    self.work_dir / "events" / source / output_folder))
        self.kernels_file = kernels_file
        p.close()

    def run_sumkernels(self, conf, output_folder, it, misfit_type):
        io.clean_mkdir(self.work_dir / output_folder)
        log, err = conf.get_log_files("sum_kernels", it)
        kernels = [k+"_kernel" for k in conf.simulation.kernels]

        preconditioners = []
        if conf.postprocessing.precond_folder is None:
            preconditioners = [p+"_kernel:absolute"
                               for p in conf.postprocessing.preconditioners]
        kernels = chain(kernels, preconditioners)

        io.run_command("sair-sum-kernels",
                       chain([conf.filename,
                              misfit_type, it,
                              self.work_dir / output_folder],
                             kernels),
                       np=self.nproc,
                       stdout=log, stderr=err,
                       on_error="Kernel Sum Error!")

        io.remove_file(self.kernels_file)

    def run_mask(self, conf, kernel_folder, it):
        log, err = conf.get_log_files("mask_kernels", it)
        kernels = [k+"_kernel" for k in conf.simulation.kernels]
        io.run_command("sair-mask-kernels",
                       chain([conf.filename, kernel_folder], kernels),
                       np=self.nproc,
                       stdout=log, stderr=err,
                       on_error="Mask Kernel Error!")

    def run_clip(self, conf, kernel_folder, it):
        log, err = conf.get_log_files("mask_kernels", it)
        io.run_command("sair-clip-kernels",
                       [conf.filename, kernel_folder],
                       np=self.nproc,
                       stdout=log, stderr=err,
                       on_error="Clip Kernel Error!")

    def run_preconditioner(self, conf, kernel_folder):
        if not conf.postprocessing.precondition:
            return False
        print("Applying preconditioner")
        work_dir = conf.root_work_dir / kernel_folder
        log, err = conf.get_log_files("precond")

        if conf.postprocessing.precond_folder:
            io.copy_files(conf.postprocessing.precond_folder + "/*",
                          work_dir)

        io.run_command("sair-precondition",
                       [work_dir, conf.filename],
                       np=self.nproc,
                       stdout=log, stderr=err,
                       on_error="Preconditioner Error!")

    def run_smooth(self, conf, kernel_folder):
        if not conf.postprocessing.smooth:
            return False
        print("Smoothing kernels")

        work_dir = conf.root_work_dir / kernel_folder
        log, err = conf.get_log_files("smooth")

        sigmas = [str(conf.postprocessing.sigmah),
                  str(conf.postprocessing.sigmav)]
        kernels = [k+"_kernel" for k in chain(conf.simulation.kernels)]
        preconditioners = []
        if conf.postprocessing.precond_folder is None:
            preconditioners = [p+"_kernel"
                               for p in conf.postprocessing.preconditioners]
        kernels = chain(kernels, preconditioners)

        options = [".", ".", ".false."]

        io.copy_folder(self.work_dir / "events" / self.sources[0] / "DATA",
                       work_dir / "DATA", clean=True)

        for kernel in kernels:
            io.run_command(self.specfem_bin / "xsmooth_sem",
                           args=chain(sigmas, [kernel], options),
                           np=self.nproc, stdout=log, stderr=err,
                           cwd=work_dir,
                           on_error="Smooth Error!")

        io.remove_folder(work_dir / "DATA")

    def run_generate_update(self, conf, misfit_type, iteration):

        log, err = conf.get_log_files("generate_update", iteration)
        io.clean_mkdir(conf.root_work_dir / "update_{}_{}".format(
            misfit_type, iteration))

        io.run_command("sair-generate-update",
                       args=[conf.filename, misfit_type, iteration],
                       np=self.nproc, stdout=log, stderr=err,
                       on_error="Update Generation Error!")

        io.copy_files(conf.root_work_dir / "kernels_{}_{}".format(
            misfit_type, iteration) / "proc*x.bin",
                      conf.root_work_dir / "update_{}_{}".format(
                      misfit_type, iteration))

        io.copy_files(conf.root_work_dir / "kernels_{}_{}".format(
            misfit_type, iteration) / "proc*z.bin",
                      conf.root_work_dir / "update_{}_{}".format(
                      misfit_type, iteration))


def run_forward_sim(source, specfem, output_folder, keep_lastframe=False):
    return specfem.run_forward_sim(source, output_folder, keep_lastframe)


def run_backward_sim(source, specfem, output_folder):
    return specfem.run_backward_sim(source, output_folder)


def run_compute_window(conf, misfit_type=None, it=None, step_no=None):
    print("Computing windows")
    log, err = conf.get_log_files("windows")
    io.run_command("sair-compute-windows",
                   np=conf.options.misfit_nproc,
                   args=[conf.filename], stdout=log, stderr=err,
                   on_error="Windows Error")


def run_ddpair(conf, misfit_type=None, it=None, step_no=None):
    print("Finding DD pairs")
    log, err = conf.get_log_files("dd_pair")
    np = conf.options.misfit_nproc if not conf.dd.event_pairing else 1
    io.run_command("sair-find-dd-pairs",
                   np=np,
                   args=[conf.filename], stdout=log, stderr=err,
                   on_error="DD Pair Error")


def run_adjoint(conf, misfit_type, calc_type, it, try_no):
    nproc = conf.options.misfit_nproc

    command = "sair-adjoint"
    if conf.simulation.seismogram_format == "su":
        command = "sair-adjoint-su"

    if not conf.options.calc_events_individually:
        log, err = conf.get_log_files("adjoint_misfit", it, try_no)
        io.run_command(command,
                       [misfit_type, calc_type, conf.filename],
                       np=nproc, stdout=log, stderr=err,
                       on_error="Adjoint Error!")
    else:
        for event in utils.tqdm(conf.get_event_names(), desc="Adjoints"):
            log, err = conf.get_log_files("adjoint_misfit_{}".format(event),
                                          it, try_no)
            io.run_command(command,
                           [misfit_type, calc_type, conf.filename, event],
                           np=nproc, stdout=log, stderr=err,
                           on_error="Adjoint Error!")


def sum_misfits(conf):
    misfit_file = conf.get_misfit_folder() / "misfits"
    if not io.is_a_file(misfit_file):
        total_misfit = 0
        for filename in io.get_files(misfit_file+"_*"):
            with open(filename) as f:
                total_misfit += float(f.readlines()[-1])
        with open(misfit_file, "w") as f:
            f.write("{:.15f}\n".format(total_misfit))
    else:
        with open(misfit_file) as f:
            total_misfit = float(f.readlines()[-1])

    return total_misfit


def run_update_model(conf, nproc, misfit_type, it, step):
    log, err = conf.get_log_files("update_model")

    io.run_command("sair-update-model",
                   args=[conf.filename, misfit_type, it, step],
                   stdout=log, stderr=err, np=nproc,
                   on_error="Model Update Error!")


def write_model_misfit(conf, new_model):
    for k in conf.simulation.kernels:
        m = conf.model_names[k]
        m_misfit = model_misfit.calc_model_misfit(
            conf.target_model, new_model, m)
        with open(conf.get_model_misfits_file(m), "a") as f:
            f.write("{:.2f}\n".format(m_misfit))


class Step(object):
    BEGIN = "BEGIN"
    TARGET_SIM = "TARGET_SIM"
    SYNT_SIM = "SYNT_SIM"
    WINDOW = "WINDOW"
    DD_PAIR = "DD_PAIR"
    ADJOINT = "ADJOINT"
    WRITE_MISFITS = "WRITE_MISFITS"
    SYNT_BACKWARD = "SYNT_BACKWARD"
    SUM_KERNEL = "SUM_KERNEL"
    MASK_KERNEL = "MASK_KERNEL"
    CLIP_KERNEL = "CLIP_KERNEL"
    SMOOTH_AROUND_SOURCES = "SMOOTH_AROUND_SOURCES"
    SMOOTH = "SMOOTH"
    PRECONDITION = "PRECONDITION"
    GENERATE_UPDATE = "GENERATE_UPDATE"
    UPDATE_MODEL = "UPDATE_MODEL"
    NEW_MODEL_SIM = "NEW_MODEL_SIM"
    NEW_MODEL_MISFIT = "NEW_MODEL_MISFIT"
    FINALIZE_ITER = "FINALIZE_ITER"
    FILE = "sair_last_step"

    def __init__(self, step_name="BEGIN", iteration=1, linesearch_no=0,
                 work_dir=None):
        self.step_name = step_name
        self.iteration = int(iteration)
        self.linesearch_no = int(linesearch_no)
        if work_dir is not None:
            self.work_dir = io.Path(work_dir)

    def is_next(self):
        return self.prev < self

    def __str__(self):
        return "Step({}, {}, {})".format(self.iteration,
                                         self.step_name,
                                         self.linesearch_no)

    def __repr__(self):
        return self.__str__()

    def update(self, other, no_write=False):
        self.iteration = other.iteration
        self.step_name = other.step_name
        self.linesearch_no = other.linesearch_no
        if not no_write:
            self.write()

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, Step):
            return self.step_name == other.step_name and \
                self.iteration == other.iteration and \
                self.linesearch_no == other.linesearch_no
        else:
            raise Exception("other is not a step.")

    @property
    def filename(self):
        return self.work_dir / Step.FILE

    def last_step_exists(self):
        return io.is_a_file(self.filename)

    def write(self):
        if not io.is_a_folder(self.work_dir):
            return
        with open(self.filename, "w") as f:
            f.write("{:s},{:d},{:d}".format(self.step_name,
                                            self.iteration,
                                            self.linesearch_no))

    def load(self):
        with open(self.filename) as f:
            a, b, c = f.read().split(",")
            return Step(a, int(b), int(c), self.work_dir)


def main():
    parser = argparse.ArgumentParser(description="Run the inversion")
    parser.add_argument("conf_file", help="config.yml file")
    parser.add_argument("misfit_type", choices=adjoint.adjs.keys(),
                        help="Misfit type")
    parser.add_argument("-y", "--always-continue",
                        help="Continue the last run without asking.",
                        action="store_true")
    parser.add_argument("-f", "--fresh-run",
                        help="Always start a new run without asking.",
                        action="store_true")
    parser.add_argument("-l", "--last-step", help="Last step",
                        default=None)
    parser.add_argument("-d", "--dry-run", help="Dry run",
                        action="store_true")
    args = parser.parse_args()
    conf = config.load_config(args.conf_file)

    misfit_type = args.misfit_type
    last_step = None
    if args.last_step:
        last_step = Step(*args.last_step.split(","))
    run(conf, misfit_type, last_step,
        args.always_continue, args.fresh_run, args.dry_run)


def prepare_target_data(conf, misfit_type, it=0, search_no=0):
    io.clean_mkdir(conf.root_work_dir)
    io.clean_mkdir(conf.get_log_folder())
    io.clean_mkdir(conf.get_misfit_folder())
    io.clean_mkdir(conf.get_model_folder())

    if conf.target_model:
        target_model = conf.get_model_folder(misfit_type, "target")
        io.copy_folder(conf.target_model, target_model, clean=True)
        target = Specfem2D(conf, target_model)
        target.run_all_forward("DATA_obs")
    else:
        # Copy target data
        for event in conf.get_event_names():
            data_obs = conf.get_obs_folder(event)
            io.clean_mkdir(data_obs)
            io.copy_folder(conf.target_data_folder / event,
                           data_obs, clean=True)

            io.clean_mkdir(conf.get_model_folder())
            first_model = conf.get_model_folder(misfit_type, 0)
            io.copy_folder(conf.init_model, first_model,
                           clean=True)

    first_model = conf.get_model_folder(misfit_type, 0)
    io.copy_folder(conf.init_model, first_model, clean=True)

    # add noise
    # TODO: Ascii support
    if conf.simulation.seismogram_format == "su":
        if conf.simulation.target_noise_level > 0:
            for comp in conf.simulation.comps:
                for event in conf.get_event_names():
                    filename = conf.get_su_filename(event, comp, "obs")
                    st = utils.read_su_file(filename)
                    utils.add_noise(st, conf.simulation.target_noise_level)
                    utils.write_su_file(st, filename)
    else:
        raise NotImplementedError("Adding noise is not supported for {} seismograms".format(
            conf.simulation.seismogram_format))


def run_synt_sym(conf, misfit_type, it, search_no=None):
    model = conf.get_model_folder(misfit_type, it)
    syn = Specfem2D(conf, model)
    syn.run_all_forward("DATA_syn", keep_lastframe=True)


def run_prev_synt_sym(conf, misfit_type, it, search_no=None):
    return run_synt_sym(conf, misfit_type, it - 1, search_no)


def run_synt_sym_backwards(conf, misfit_type, it, search_no=None):
    prev_model = conf.get_model_folder(misfit_type, it - 1)
    syn = Specfem2D(conf, prev_model)
    kernels_name = "kernels_{}_{}".format(misfit_type, it)
    syn.run_all_backward(kernels_name)


def sum_kernels(conf, misfit_type, it, search_no=None):
    prev_model = conf.get_model_folder(misfit_type, it - 1)
    syn = Specfem2D(conf, prev_model)

    # If there is no kernels_file, create it.
    if not io.is_a_file(syn.kernels_file):
        with open(syn.kernels_file, "w") as f:
            for event in conf.get_event_names():
                f.write("{}\n".format(
                    conf.get_event_kernel_folder(event, misfit_type, it)))

    print("Summing kernels")
    kernels_name = "kernels_{}_{}".format(misfit_type, it)
    syn.run_sumkernels(conf, kernels_name, it, misfit_type)

    # These are needed for smoothing
    io.copy_files(prev_model / "proc*x.bin",
                  conf.root_work_dir / kernels_name)
    io.copy_files(prev_model / "proc*z.bin",
                  conf.root_work_dir / kernels_name)
    io.copy_files(prev_model / "proc*NSPEC_ibool.bin",
                  conf.root_work_dir / kernels_name)
    io.copy_files(prev_model / "proc*jacobian.bin",
                  conf.root_work_dir / kernels_name)


def mask_kernels(conf, misfit_type, it, search_no=None):
    if conf.postprocessing.mask_sources or conf.postprocessing.mask_stations:
        prev_model = conf.get_model_folder(misfit_type, it - 1)
        syn = Specfem2D(conf, prev_model)
        print("Masking sources/stations")
        kernels_folder = "kernels_{}_{}".format(misfit_type, it)
        syn.run_mask(conf, conf.root_work_dir / kernels_folder, it)
    else:
        return False


def clip_kernels(conf, misfit_type, it, search_no=None):
    if conf.postprocessing.clip_kernels:
        prev_model = conf.get_model_folder(misfit_type, it - 1)
        syn = Specfem2D(conf, prev_model)
        print("Clipping kernels")
        kernels_folder = "kernels_{}_{}".format(misfit_type, it)
        syn.run_clip(conf, conf.root_work_dir / kernels_folder, it)
    else:
        return False


def smooth_around_sources(conf, misfit_type, it, search_no=None):
    if conf.postprocessing.smooth_around_sources or \
       conf.postprocessing.smooth_around_stations:
        print("Smoothing around sources")
        prev_model = conf.get_model_folder(misfit_type, it - 1)
        syn = Specfem2D(conf, prev_model)
        log, err = conf.get_log_files("smooth_around_sources", it, search_no)
        kernels_name = conf.root_work_dir / "kernels_{}_{}".format(misfit_type, it)  # NOQA
        io.run_command("sair-smooth-around-sources",
                       [conf.filename, kernels_name],
                       np=syn.nproc, stdout=log, stderr=err,
                       on_error="Smooth around sources Error!")


def run_smooth(conf, misfit_type, it, search_no=None):
    prev_model = conf.get_model_folder(misfit_type, it - 1)
    syn = Specfem2D(conf, prev_model)
    if conf.postprocessing.use_specfem_xsmooth:
        kernels_name = "kernels_{}_{}".format(misfit_type, it)
        syn.run_smooth(conf, kernels_name)
    elif conf.postprocessing.smooth or conf.postprocessing.smooth_adaptive:
        print("Smoothing kernels (python)")
        kernels_name = conf.root_work_dir / "kernels_{}_{}".format(misfit_type, it)  # NOQA
        log, err = conf.get_log_files("smooth", it, search_no)
        io.run_command("sair-smooth-kernels",
                       [conf.filename, kernels_name],
                       np=syn.nproc, stdout=log, stderr=err,
                       on_error="Smooth Error!")


def run_precondition(conf, misfit_type, it, search_no=None):
    prev_model = conf.get_model_folder(misfit_type, it - 1)
    syn = Specfem2D(conf, prev_model)
    kernels_name = "kernels_{}_{}".format(misfit_type, it)
    syn.run_preconditioner(conf, kernels_name)


def run_gen_update(conf, misfit_type, it, search_no=None):
    print("Generating Update")
    prev_model = conf.get_model_folder(misfit_type, it - 1)
    syn = Specfem2D(conf, prev_model)
    syn.run_generate_update(conf, misfit_type, it)


def run_model_update(conf, misfit_type, it, search_no):
    linesearch_file = conf.get_linesearch_file(misfit_type, it)
    alpha = linesearch.get_next_step(conf, linesearch_file)
    prev_model = conf.get_model_folder(misfit_type, it - 1)
    syn = Specfem2D(conf, prev_model)
    print("step: ", alpha)
    print("Running Model Update")
    run_update_model(conf, syn.nproc, misfit_type, it, alpha)


def calc_adjoint_sources(conf, misfit_type, it, search_no):
    print("Running misfit")
    io.clean_mkdir(conf.get_misfit_folder())
    run_adjoint(conf, misfit_type, "both", it, search_no)
    misfit = sum_misfits(conf)
    print("Misfit: {:.10f}".format(misfit))
    with open(conf.get_linesearch_file(
            misfit_type, it), "w") as f:
        f.write("{:.5f} {:.15f}\n".format(0.0, misfit))


def run_misfit(conf, misfit_type, it, search_no):
    print("Calculating misfit")
    io.clean_mkdir(conf.get_misfit_folder())
    run_adjoint(conf, misfit_type, "misfit", it, search_no)
    misfit = sum_misfits(conf)
    linesearch_file = conf.get_linesearch_file(misfit_type, it)
    alpha = linesearch.get_next_step(conf, linesearch_file)
    print("{:.5f} {:.10f}".format(alpha, misfit))
    with open(linesearch_file, "a") as f:
        f.write("{:.5f} {:.15f}\n".format(alpha, misfit))


def collect_obs_traces(conf, misfit_type, it, search_no=None):
    if conf.keep_data.traces:
        data_folder = conf.root_work_dir / "DATA"
        iter_folder = data_folder / "iter_{}".format(it)
        io.makedir(iter_folder)
        for event in conf.get_event_names():
            io.copy_folder(conf.get_obs_folder(event),
                           conf.get_collected_trace_folder(event, it, "obs"),
                           clean=True)


def collect_syn_traces(conf, misfit_type, it, search_no=None):
    if conf.keep_data.traces:
        data_folder = conf.root_work_dir / "DATA"
        iter_folder = data_folder / "iter_{}".format(it)
        io.makedir(iter_folder)
        for event in conf.get_event_names():
            io.copy_folder(conf.get_syn_folder(event),
                           conf.get_collected_trace_folder(event, it, "syn"),
                           clean=True)


def collect_adjoint_sources(conf, misfit_type, it, search_no=None):
    if conf.keep_data.adjoint_sources:
        data_folder = conf.root_work_dir / "DATA"
        iter_folder = data_folder / "iter_{}".format(it)
        io.makedir(iter_folder)
        for event in conf.get_event_names():
            io.copy_folder(
                conf.get_sem_folder(event),
                conf.get_collected_trace_folder(event, it, "adjoint"),
                clean=True)


def collect_event_kernels(conf, misfit_type, it, search_no=None):
    if conf.keep_data.event_kernels:
        data_folder = conf.root_work_dir / "DATA"
        iter_folder = data_folder / "iter_{}".format(it)
        io.makedir(iter_folder)
        for event in conf.get_event_names():
            io.copy_folder(
                conf.get_event_kernel_folder(event, misfit_type, it),
                conf.get_collected_event_kernel_folder(event, it),
                clean=True)


def write_misfits(conf, misfit_type, it, search_no=None):
    prev_model = conf.get_model_folder(misfit_type, it - 1)
    misfit = sum_misfits(conf)
    with open(conf.get_data_misfits_file(), "a") as f:
        f.write("{:.15f}\n".format(misfit))
    if conf.target_model:
        write_model_misfit(conf, prev_model)


def finalize_iter(conf, misfit_type, it, search_no=None):
    linesearch_file = conf.get_linesearch_file(misfit_type, it)
    best_step, best_misfit = linesearch.get_best(conf, linesearch_file)
    print("best_step: ", best_step)
    model = conf.get_model_folder(misfit_type, it)
    syn = Specfem2D(conf, model)
    run_update_model(conf, syn.nproc, misfit_type, it, best_step)

    with open(conf.get_data_misfits_file(), "a") as f:
        f.write("{:.15f}\n".format(best_misfit))

    if conf.target_model:
        write_model_misfit(conf, model)


def copy_config(conf, misfit_type=None, it=None, search_no=None):
    io.copy_file(conf.filename,
                 conf.root_work_dir / "config.yml")


def copy_data(conf, misfit_type=None, it=None, search_no=None):
    data_folder = conf.root_work_dir / "DATA"
    io.clean_mkdir(data_folder)
    io.copy_file(conf.sources_file, data_folder / "events")
    io.copy_folder(conf.station_folder, data_folder / "stations",
                   clean=True)
    io.copy_folder(conf.source_folder, data_folder / "sources",
                   clean=True)
    adj = conf.adjoint
    if adj.receiver_weights:
        io.copy_folder(adj.receiver_weights,
                       data_folder / "receiver_weights")
    if adj.source_weights:
        io.copy_file(adj.source_weights,
                     data_folder / "source_weights")


def inversion_recipe(conf, misfit_type):
    yield Step(Step.BEGIN), None
    for it in range(conf.inversion.start_it, conf.inversion.stop_it+1):
        yield Step("PRINT_ITER_NUM", it), \
            lambda x, y, z, t: print("Iteration {}".format(z))
        if it == 1:
            yield Step(Step.TARGET_SIM, 1), prepare_target_data
            yield Step("COPY_CONFIG"), copy_config
            yield Step("COPY_DATA"), copy_data
            yield Step("COLLECT_OBS_TRACES"), collect_obs_traces

        for search_no in range(conf.linesearch.ntry):
            if search_no == 0:

                yield Step(Step.SYNT_SIM, it, 0), run_prev_synt_sym
                yield Step("COLLECT_SYN_TRACES"), collect_syn_traces

                if it == 1:
                    yield Step(Step.WINDOW, it, 0), run_compute_window

                if it == 1 and "dd" in misfit_type:
                    yield Step(Step.DD_PAIR, 1, 0), run_ddpair

                yield Step(Step.ADJOINT, it, search_no), calc_adjoint_sources
                yield Step("COLLECT_ADJ_SOURCES"), collect_adjoint_sources

                if it == 1:
                    yield (Step(Step.WRITE_MISFITS, it, search_no),
                           write_misfits)

                yield (Step(Step.SYNT_BACKWARD, it, search_no),
                       run_synt_sym_backwards)
                yield Step("COLLECT_EVENT_KERNELS"), collect_event_kernels

                yield (Step(Step.SUM_KERNEL, it, search_no),
                       sum_kernels)

                yield (Step(Step.MASK_KERNEL, it, search_no),
                       mask_kernels)
                yield (Step(Step.CLIP_KERNEL, it, search_no),
                       clip_kernels)
                yield (Step(Step.SMOOTH_AROUND_SOURCES, it, search_no),
                       smooth_around_sources)

                yield Step(Step.SMOOTH, it, search_no), run_smooth
                yield Step(Step.PRECONDITION, it, search_no), run_precondition
                yield Step(Step.GENERATE_UPDATE, it, search_no), run_gen_update

            yield Step(Step.UPDATE_MODEL, it, search_no), run_model_update
            yield Step(Step.NEW_MODEL_SIM, it, search_no), run_synt_sym
            yield Step(Step.NEW_MODEL_MISFIT, it, search_no), run_misfit

        yield Step(Step.FINALIZE_ITER, it), finalize_iter


def run(conf, misfit_type, last_step=None,
        should_continue=False, fresh_run=False,
        dry_run=False):
    cur_step = Step(work_dir=conf.root_work_dir)

    if not fresh_run:
        if cur_step.last_step_exists():
            if not should_continue:
                ans = raw_input("Continue the last run [y/n]: ")
                if ans == "y":
                    should_continue = True
            if should_continue:
                print("Continuing")
                cur_step = cur_step.load()

    caught_up = False
    for step, func in inversion_recipe(conf, misfit_type):
        if not caught_up and step == cur_step:
            caught_up = True
            continue
        try:
            if caught_up:
                if dry_run:
                    print(cur_step)
                else:
                    func(conf, misfit_type,
                         step.iteration,
                         step.linesearch_no)
                cur_step.update(step, no_write=dry_run)
                if cur_step == last_step:
                    if dry_run:
                        print(cur_step)
                    break

        except linesearch.LinesearchFailed:
            print("Couldn't reduce the misfit")
            break

    # TODO: Decide if we want to keep finished works, labeled as finished.
    # io.remove_file(cur_step.filename)


def multiple_run(run_file, last_step=None,
                 should_continue=False, fresh_run=False,
                 dry_run=False):
    for name, conf, misfit_type in config.get_runs(run_file):
        print("Running {}".format(name))
        run(conf, misfit_type, last_step, should_continue,
            fresh_run, dry_run)


def multiple_run_command():
    parser = argparse.ArgumentParser(description="Run multiple inversions")
    parser.add_argument("run_file", help="run file")
    parser.add_argument("-y", "--always-continue",
                        help="Continue the last run without asking.",
                        action="store_true")
    parser.add_argument("-f", "--fresh-run",
                        help="Always start a new run without asking.",
                        action="store_true")
    parser.add_argument("-l", "--last-step", help="Last step",
                        default=None)
    parser.add_argument("-d", "--dry-run", help="Dry run",
                        action="store_true")

    args = parser.parse_args()
    multiple_run(args.run_file,
                 args.last_step,
                 args.always_continue,
                 args.fresh_run,
                 args.dry_run)


# TODO: Convert this to a recipe?
def prepare_models():
    parser = argparse.ArgumentParser(description="Run the inversion")
    parser.add_argument("conf_file", help="config.yml file")
    args = parser.parse_args()
    conf = config.load_config(args.conf_file)
    if conf.prepare_model is None:
        raise Exception("Config does not have a prepare script.")

    sources = conf.get_event_names()
    source = sources[0]

    io.clean_mkdir(conf.root_work_dir)
    io.clean_mkdir(conf.get_log_folder())

    target = Specfem2D(conf, conf.target_model)

    target.set_specfem_vars({
        "NPROC": conf.prepare_model.nproc,
        "MODEL": "default",
        "SAVE_MODEL": "binary"
    })
    work_dir, data_dir, logs_dir = target._prepare(source)
    io.remove_files(data_dir / "*.bin")
    target.run_mesher(work_dir, logs_dir)
    target.run_specfem(work_dir, logs_dir)

    target.set_specfem_vars({
        "NPROC": conf.prepare_model.nproc,
        "MODEL": "binary",
        "SAVE_MODEL": "default"
    })
    target._apply_variables(data_dir)

    if conf.prepare_model.target_command:
        commandp = conf.prepare_model.target_command.split()
        command = commandp[0]
        command_args = chain(commandp[1:], [data_dir])
        np = conf.prepare_model.nproc if conf.prepare_model.mpi else 1
        io.run_command(command, command_args, np=np,
                       stdout=io.Path("work/logs/prepare_target_model_log"),
                       stderr=io.Path("work/logs/prepare_target_model_err"),
                       on_error="Target model error!")

    # Copy new empty model to new location
    io.remove_file(data_dir / "SOURCE")
    io.remove_file(data_dir / "STATIONS")
    io.copy_folder(data_dir, conf.target_model + "_new", clean=True)

    init = Specfem2D(conf, conf.init_model)

    init.set_specfem_vars({
        "NPROC": conf.prepare_model.nproc,
        "MODEL": "default",
        "SAVE_MODEL": "binary"
    })

    work_dir, data_dir, logs_dir = init._prepare(source)
    io.remove_files(data_dir / "*.bin")
    init.run_mesher(work_dir, logs_dir)
    init.run_specfem(work_dir, logs_dir)

    init.set_specfem_vars({
        "NPROC": conf.prepare_model.nproc,
        "MODEL": "binary",
        "SAVE_MODEL": "default"
    })
    init._apply_variables(data_dir)

    if conf.prepare_model.init_command:
        commandp = conf.prepare_model.init_command.split()
        command = commandp[0]
        command_args = chain(commandp[1:], [data_dir])
        np = conf.prepare_model.nproc if conf.prepare_model.mpi else 1
        io.run_command(command, command_args, np=np,
                       stdout=io.Path("work/logs/prepare_init_model_log"),
                       stderr=io.Path("work/logs/prepare_init_model_err"),
                       on_error="Init model error!")

    # Copy new empty model to new location
    io.remove_file(data_dir / "SOURCE")
    io.remove_file(data_dir / "STATIONS")
    io.copy_folder(data_dir, conf.init_model + "_new", clean=True)


if __name__ == "__main__":
    main()
