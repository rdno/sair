def calc_convolved(calc_type, events, conf):
    if mpi.is_main_rank:
        pairs = {"X": [], "Y": [], "Z": []}
        for event in conf.get_event_names():
            for comp in conf.simulation.comps:
                c = comp[-1]
                ref = None
                for sta in conf.get_station_list(event):
                    staname = ".".join([sta["net"], sta["name"], comp, event]) + ":0"  # NOQA
                    if ref is None:
                        ref = staname
                    pairs[c].append({
                        "window_id_i": staname,
                        "weight_i": 1.0,
                        "window_id_j": "00.0000.{}.{}:0".format(comp, event),
                        "weight_j": 1.0
                    })
        import json
        with open(conf.root_work_dir / "pairs.json", "w") as f:
            json.dump(pairs, f, indent=2, sort_keys=True)
        print("NEW PAIRS!")
    mpi.comm.barrier()
    adjoint_config = ConfigDoubleDifferenceWaveForm(
        1.0/conf.adjoint.freq_max, 1.0/conf.adjoint.freq_min,
        conf.adjoint.taper_type, conf.adjoint.taper_percentage)
    adjoint_misfit_type = "convolved_wavefield"

    all_pairs = dd_pair.read_pairs(conf, events)
    stanames, obsds, synts, windows, _ = read_pair_files(conf, events,
                                                         all_pairs)

    dd_weights = weights.DDWeights(all_pairs,
                                   conf.dd.pair_wise_weighting)
    rec_weights = weights.ReceiverWeights(conf, events)
    rec_weights.find_norm_factors(dd_weights,
                                  cc_windows={})
    if mpi.is_main_rank:
        rec_weights.analyze(dd_weights, {})

    calc = partial(single_dd_misfit,
                   pairs=all_pairs,
                   obsds=obsds,
                   synts=synts,
                   windows=windows,
                   adjoint_misfit_type=adjoint_misfit_type,
                   adjoint_config=adjoint_config,
                   calc_type=calc_type,
                   conf=conf,
                   dd_weights=dd_weights,
                   rec_weights=rec_weights)

    results = mpi.run_with_mpi(calc, list(range(len(all_pairs))),
                               title="DD")

    if calc_type & CALC_MISFIT:
        def write_misfits():
            total_misfit = 0
            misfits = defaultdict(float)
            for stap, v in results.iteritems():
                sta, p = stap.split(":")
                net, sta, comp, event = sta.split(".")
                # TODO: Station based misfit
                staname = ".".join([net, sta, comp])
                total_misfit += v["misfit"]
            filename = conf.get_misfit_folder() / "misfits"
            if len(events) == 1:
                filename += "_" + events[0]
            with open(filename, "w") as f:
                f.write("{:.15f}\n".format(total_misfit))
        mpi.comm.barrier()
        mpi.run_on_main_rank(write_misfits)

    if calc_type & CALC_ADJOINT:
        def write_srcs():
            times = obsds[stanames[0]].times()
            dt = times[1] - times[0]
            adjs = {sta: np.zeros(len(times)) for sta in stanames}

            # Sum over pairs
            for stap, adj in tqdm(results.iteritems()):
                sta, p = stap.split(":")
                adjs[sta] += adj["src"]

            utils.write_su_adjoint(conf, adjs, dt, events)

        mpi.comm.barrier()
        mpi.run_on_main_rank(write_srcs)
