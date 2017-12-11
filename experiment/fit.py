"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

usage:
$ python fit.py [-h] [--J P] [--D P] [--npg P [P ...]] [--cor_input B]
                [--run_all B] [--run_ep B] [--run_full B] [--run_consensus B]
                [--run_target B] [--iter P] [--siter P] [--target_siter P]
                [--chains P] [--K P] [--damp F] [--mix B] [--prec_estim S]
                [--seed_data N] [--seed_mcmc N] [--id S] [--save_true B]
                [--save_res B] [--save_target_samp B]
                model_name

positional arguments:
  model_name            name of the model

optional arguments - general:
  -h, --help            show this help message and exit

optional arguments - data:
  --J P                 number of hierarchical groups, default 40
  --D P                 number of inputs, default 20
  --npg P [P ...]       number of observations per group (constant or min
                        max), default 20
  --cor_input B         correlated input variable, default False

optional arguments - selected methods:
  --run_all B           run all the methods, default False
  --run_ep B            run the distributed EP method, default False
  --run_full B          run the full model method, default False
  --run_consensus B     run consensus MC method, default False
  --run_target B        run target approximation, default False

optional arguments - iterations:
  --iter P              number of distributed EP iterations, default 6
  --siter P             Stan iterations in each major iteration, default 400
  --target_siter P      Stan iterations for the target approximation, default
                        1000
  --chains P            number of chains used in stan sampling, default 4

optional arguments - method options:
  --K P                 number of sites, default 25
  --damp F              damping factor constant, default 0.75
  --mix B               mix last iteration samples, default False
  --prec_estim S        estimate method for tilted distribution precision
                        matrix, currently available options are sample and
                        olse (see epstan.method.Master), default sample

optional arguments - seeds for randomisation:
  --seed_data N         seed for data simulation, default 0
  --seed_mcmc N         seed for sampling, default 0

optional arguments - saving options:
  --id S                optional id appended to the end of the result files,
                        default None
  --save_true B         save true values, default True
  --save_res B          save results, default True
  --save_target_samp B  save target approximation samples, default False

Argument types
- N denotes a non-negative and P a positive integer argument.
- F denotes a float argument
- B denotes a boolean argument, which can be given as
  TRUE, T, 1 or FALSE, F, 0 (case insensitive).
- S denotes a string argument.

Available models are defined in the folder models in the files
`<model_name>.py`, `<model_name>.stan` and `<model_name>_sg.stan`

The results of full model method are saved into file
    `res_f_<model_name>.npz`,
the results of distributed method are saved into file
    `res_d_<model_name>.npz`
the results of consensus MC method are saved into file
    `res_c_<model_name>.npz`
and the true values are saved into the file
    `true_vals_<model_name>.npz`
into the folder results.

After running this skript for all the methods, the script plot_res.py can be
used to plot the results.

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import os
import argparse

import numpy as np
from scipy import linalg

import pystan


# Add parent dir to sys.path if not present already. This is only done because
# of easy importing of the package epstan. Adding the parent directory into the
# PYTHONPATH works as well.
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(CUR_PATH, os.pardir))
RES_PATH = os.path.join(CUR_PATH, 'results')
MOD_PATH = os.path.join(CUR_PATH, 'models')
# Double check that the package is in the parent directory
if os.path.exists(os.path.join(PARENT_PATH, 'epstan')):
    if PARENT_PATH not in os.sys.path:
        os.sys.path.insert(0, PARENT_PATH)

from epstan.method import Master
from epstan.util import (
    load_stan, distribute_groups, stan_sample_time, stan_sample_subprocess)


CONFS = [
    'J', 'D', 'npg', 'cor_input',
    'run_all', 'run_ep', 'run_full', 'run_consensus', 'run_target',
    'iter', 'siter', 'target_siter', 'chains',
    'K', 'damp', 'mix', 'prec_estim',
    'seed_data', 'seed_mcmc',
    'id', 'save_true', 'save_res', 'save_target_samp',
]

CONF_DEFAULT = dict(

    J                = 20,
    D                = 16,
    K                = 10,
    npg              = 20,
    cor_input        = True,

    run_all          = False,
    run_ep           = False,
    run_full         = False,
    run_consensus    = False,
    run_target       = False,

    iter             = 6,
    siter            = 400,
    target_siter     = 10000,
    chains           = 4,

    damp             = 0.8,
    mix              = False,
    prec_estim       = 'sample',

    seed_data        = 0,
    seed_mcmc        = 0,

    id               = None,
    save_true        = True,
    save_res         = True,
    save_target_samp = False,

)

FULL_ITERS = [50, 100, 200, 400, 800, 1600, 3200]
CONS_ITERS = [50, 100, 500, 1000, 1500, 2000]


class configurations(object):
    """Configuration container for the function main."""
    def __init__(self, **kwargs):
        # Set given options
        for k, v in kwargs.items():
            if k not in CONF_DEFAULT:
                raise ValueError("Invalid option `{}`".format(k))
            setattr(self, k, v)
        # Set missing options to defaults
        for k, v in CONF_DEFAULT.items():
            if k not in kwargs:
                setattr(self, k, v)
    def __str__(self):
        conf_dict = self.__dict__
        opts = ['{!s} = {!r}'.format(opt, conf_dict[opt])
                for opt in CONFS if opt in conf_dict]
        return '\n'.join(opts)
    def __repr__(self):
        return self.__str__()


def main(model_name, conf, ret_master=False):
    """Fit requested model with given configurations.

    Arg. `ret_master` can be used to prematurely exit and return the
    epstan.Master object, which is useful for debuging.

    """

    # Ensure that the configurations class is used
    if not isinstance(conf, configurations):
        raise ValueError("Invalid arg. `conf`, use class fit.configurations")

    print("Configurations:")
    print('    ' + str(conf).replace('\n', '\n    '))

    # Localise few options
    J = conf.J
    D = conf.D
    K = conf.K

    # Import the model simulator module (import at runtime)
    model_module = getattr(__import__('models.'+model_name), model_name)
    model = model_module.model(J, D, conf.npg)

    # Simulate_data
    if conf.cor_input:
        data = model.simulate_data(Sigma_x='rand', rng=conf.seed_data)
    else:
        data = model.simulate_data(rng=conf.seed_data)

    # Calculate the uncertainty
    uncertainty_global, uncertainty_group = data.calc_uncertainty()

    # Get the prior
    S0, m0, Q0, r0 = model.get_prior()
    prior = {'Q':Q0, 'r':r0}

    #~ # Set init_site to N(0,A**2/K I), where A = 10 * max(diag(S0))
    #~ init_site = 10 * np.max(np.diag(S0))
    init_site = None # Zero initialise the sites

    # Get parameter information
    pnames, pshapes, phiers = model.get_param_definitions()

    # Save true values
    if conf.save_true:
        if not os.path.exists(RES_PATH):
            os.makedirs(RES_PATH)
        if conf.id:
            filename = 'true_vals_{}_{}.npz'.format(model_name, conf.id)
        else:
            filename = 'true_vals_{}.npz'.format(model_name)
        np.savez(
            os.path.join(RES_PATH, filename),
            J = J,
            D = D,
            npg = conf.npg,
            seed = conf.seed_data,
            pnames = pnames,
            uncertainty_global = uncertainty_global,
            uncertainty_group = uncertainty_group,
            X_param = data.X_param,
            **data.true_values
        )
        print("True values saved into results")

    # --------------------------------------------------------------------------
    #   Distributed method
    # --------------------------------------------------------------------------
    if conf.run_ep or conf.run_all or ret_master:

        print("Distributed method")

        # Options for the ep-algorithm see documentation of epstan.method.Master
        epstan_options = dict(
            prior = prior,
            seed = conf.seed_mcmc,
            prec_estim = conf.prec_estim,
            df0 = conf.damp,
            init_site = init_site,
            chains = conf.chains,
            iter = conf.siter,
            warmup = None,
            thin = 1,
        )

        if K < 2:
            raise ValueError("K should be at least 2.")

        elif K < J:
            # ------ Many groups per site: combine groups ------
            Nk, Nj_k, j_ind_k = distribute_groups(J, K, data.Nj)
            # Create the Master instance
            epstan_master = Master(
                os.path.join(MOD_PATH, model_name),
                data.X,
                data.y,
                A_k = {'J':Nj_k},
                A_n = {'j_ind':j_ind_k+1},
                site_sizes = Nk,
                **epstan_options
            )
            # Construct the map: which site contribute to which parameter
            pmaps = _create_pmaps(phiers, J, K, Nj_k)

        elif K == J:
            # ------ One group per site ------
            # Create the Master instance
            epstan_master = Master(
                os.path.join(MOD_PATH, model_name+'_sg'),
                data.X,
                data.y,
                site_sizes=data.Nj,
                **epstan_options
            )
            # Construct the map: which site contribute to which parameter
            pmaps = _create_pmaps(phiers, J, K, None)

        elif K <= data.N:
            # ------ Multiple sites per group: split groups ------
            raise NotImplementedError("Splitting the groups not implemented.")

        else:
            raise ValueError("K cant be greater than number of samples")

        if ret_master:
            print("Returning epstan.Master")
            return epstan_master

        # initial approximation
        S_ep_init, m_ep_init = epstan_master.cur_approx()

        # Run the algorithm for `EP_ITER` iterations
        print(
            "Run distributed EP algorithm for {} iterations."
            .format(conf.iter)
        )
        if conf.mix:
            info, (m_s_ep, S_s_ep), (time_s_ep, mstepsize_s_ep, mrhat_s_ep) = (
                epstan_master.run(
                    conf.iter, return_analytics=True, save_last_param=pnames)
            )
        else:
            info, (m_s_ep, S_s_ep), (time_s_ep, mstepsize_s_ep, mrhat_s_ep) = (
                epstan_master.run(conf.iter, return_analytics=True)
            )

        # cumulate elapsed time in the sampling runtime analysis
        time_s_ep = time_s_ep.cumsum()

        # add initial approx info
        S_s_ep = np.concatenate((S_ep_init[None,:,:], S_s_ep), axis=0)
        m_s_ep = np.concatenate((m_ep_init[None,:], m_s_ep), axis=0)
        time_s_ep = np.insert(time_s_ep, 0, 0.0)
        mstepsize_s_ep = np.insert(mstepsize_s_ep, 0, np.nan)
        mrhat_s_ep = np.insert(mrhat_s_ep, 0, np.nan)

        # check if run failed
        if info:
            # Save results until failure
            if conf.save_res:
                if not os.path.exists(RES_PATH):
                    os.makedirs(RES_PATH)
                if conf.id:
                    filename = 'res_d_{}_{}.npz'.format(model_name, conf.id)
                else:
                    filename = 'res_d_{}.npz'.format(model_name)
                np.savez(
                    os.path.join(RES_PATH, filename),
                    conf = conf.__dict__,
                    m_s_ep = m_s_ep,
                    S_s_ep = S_s_ep,
                    time_s_ep = time_s_ep,
                    mstepsize_s_ep = mstepsize_s_ep,
                    mrhat_s_ep = mrhat_s_ep,
                    last_iter = epstan_master.iter
                )
                print("Uncomplete distributed model results saved.")
            raise RuntimeError(
                'epstan algorithm failed with error code: {}'
                .format(info)
            )

        if conf.mix:
            print("Form the final approximation "
                  "by mixing the last samples from all the sites.")
            S_ep, m_ep = epstan_master.mix_phi()

            # Get mean and var of inferred variables
            pms, pvars = epstan_master.mix_pred(pnames, pmaps, pshapes)
            # Construct a dict of from these results
            presults = {}
            for i in range(len(pnames)):
                pname = pnames[i]
                presults['m_'+pname+'_ep'] = pms[i]
                presults['v_'+pname+'_ep'] = pvars[i]

        # Save results
        if conf.save_res:
            if not os.path.exists(RES_PATH):
                os.makedirs(RES_PATH)
            if conf.id:
                filename = 'res_d_{}_{}.npz'.format(model_name, conf.id)
            else:
                filename = 'res_d_{}.npz'.format(model_name)
            if conf.mix:
                np.savez(
                    os.path.join(RES_PATH, filename),
                    conf = conf.__dict__,
                    m_s_ep = m_s_ep,
                    S_s_ep = S_s_ep,
                    time_s_ep = time_s_ep,
                    mstepsize_s_ep = mstepsize_s_ep,
                    mrhat_s_ep = mrhat_s_ep,
                    m_phi_ep = m_ep,
                    S_phi_ep = S_ep,
                    **presults
                )
            else:
                np.savez(
                    os.path.join(RES_PATH, filename),
                    conf = conf.__dict__,
                    m_s_ep = m_s_ep,
                    S_s_ep = S_s_ep,
                    time_s_ep = time_s_ep,
                    mstepsize_s_ep = mstepsize_s_ep,
                    mrhat_s_ep = mrhat_s_ep,
                )
            print("Distributed model results saved.")

        # Release master object
        del epstan_master

        print("Done with distributed method")

    # --------------------------------------------------------------------------
    #   Full model sampling
    # --------------------------------------------------------------------------
    if conf.run_full or conf.run_all:

        print("Full model")

        data_full = dict(
            N = data.X.shape[0],
            D = data.X.shape[1],
            J = J,
            X = data.X,
            y = data.y,
            j_ind = data.j_ind+1,
            mu_phi = m0,
            Omega_phi = Q0.T    # Q0 transposed in order to get C-contiguous
        )

        # sample multiple times with different number of iterations
        # preallocate output arrays
        m_s_full = np.full((len(FULL_ITERS), model.dphi), np.nan)
        S_s_full = np.full((len(FULL_ITERS), model.dphi, model.dphi), np.nan)
        time_s_full = np.full(len(FULL_ITERS), np.nan)
        mstepsize_s_full = np.full(len(FULL_ITERS), np.nan)
        mrhat_s_full = np.full(len(FULL_ITERS), np.nan)
        for i, iters in enumerate(FULL_ITERS):

            print('  iter {}: {}'.format(i+1, iters))

            # use same seed for each iteration
            seed = np.random.RandomState(seed=conf.seed_mcmc)

            # Sample and extract samples
            (samples, max_sampling_time, mean_stepsize, max_rhat, _
            ) = stan_sample_subprocess(
                model = os.path.join(MOD_PATH, model_name),
                pars = 'phi',
                data = data_full,
                seed = seed,
                chains = conf.chains,
                iter = iters,
                thin = 1
            )
            time_s_full[i] = max_sampling_time
            mstepsize_s_full[i] = mean_stepsize
            mrhat_s_full[i] = max_rhat
            samples = samples['phi']

            # Moment estimates
            nsamp = samples.shape[0]
            samples.mean(axis=0, out=m_s_full[i])
            samples -= m_s_full[i]
            samples.T.dot(samples, out=S_s_full[i])
            S_s_full[i] /= nsamp - 1

        # Save results
        if conf.save_res:
            if not os.path.exists(RES_PATH):
                os.makedirs(RES_PATH)
            if conf.id:
                filename = 'res_f_{}_{}.npz'.format(model_name, conf.id)
            else:
                filename = 'res_f_{}.npz'.format(model_name)
            np.savez(
                os.path.join(RES_PATH, filename),
                conf = conf.__dict__,
                m_s_full = m_s_full,
                S_s_full = S_s_full,
                time_s_full = time_s_full,
                mstepsize_s_full = mstepsize_s_full,
                mrhat_s_full = mrhat_s_full,
            )
            print("Full model results saved.")

        print("Done with full model method")

    # --------------------------------------------------------------------------
    #   Consensus MC
    # --------------------------------------------------------------------------
    if conf.run_consensus or conf.run_all:

        print("Consensus MC")

        seed = np.random.RandomState(seed=conf.seed_mcmc)

        # prior for each consensus site
        cons_prior_k_Q = Q0 / K
        cons_prior_k_r = r0 / K
        cons_prior_k_m = linalg.cho_solve(
            linalg.cho_factor(cons_prior_k_Q),
            cons_prior_k_r
        )

        if K < 2:
            raise ValueError("K should be at least 2.")

        elif K < J:
            # ------ Many groups per site: combine groups ------
            stan_model_name = os.path.join(MOD_PATH, model_name)
            # generate datas for each site
            Nk, Nj_k, j_ind_k = distribute_groups(J, K, data.Nj)
            k_lim = np.concatenate(([0], np.cumsum(Nk)))
            data_k = tuple(
                dict(
                    N = Nk[k],
                    D = D,
                    J = Nj_k[k],
                    X = data.X[k_lim[k]:k_lim[k+1]],
                    y = data.y[k_lim[k]:k_lim[k+1]],
                    j_ind = j_ind_k[k_lim[k]:k_lim[k+1]] + 1,
                    mu_phi = cons_prior_k_m,
                    Omega_phi = cons_prior_k_Q
                )
                for k in range(K)
            )

        elif K == J:
            # ------ One group per site ------
            stan_model_name = os.path.join(MOD_PATH, model_name+'_sg')
            # generate datas for each site
            data_k = tuple(
                dict(
                    N = data.Nj[k],
                    D = D,
                    X = data.X[data.j_lim[k]:data.j_lim[k+1]],
                    y = data.y[data.j_lim[k]:data.j_lim[k+1]],
                    mu_phi = cons_prior_k_m,
                    Omega_phi = cons_prior_k_Q
                )
                for k in range(K)
            )

        elif K <= data.N:
            # ------ Multiple sites per group: split groups ------
            raise NotImplementedError("Splitting the groups not implemented.")

        else:
            raise ValueError("K cant be greater than number of samples")

        # sample multiple times with different number of iterations
        # determine seeds for each site, constant for each iteration
        seeds = (
            np.random.RandomState(seed=conf.seed_mcmc)
            .randint(0, pystan.constants.MAX_UINT, size=K)
        )
        # preallocate output arrays
        m_s_cons = np.full((len(CONS_ITERS), model.dphi), np.nan)
        S_s_cons = np.full((len(CONS_ITERS), model.dphi, model.dphi), np.nan)
        time_s_cons = np.full(len(CONS_ITERS), np.nan)
        mstepsize_s_cons = np.full(len(CONS_ITERS), np.nan)
        mrhat_s_cons = np.full(len(CONS_ITERS), np.nan)
        for i, iters in enumerate(CONS_ITERS):

            print('  iter {}: {}'.format(i+1, iters))

            # sample for each site
            samples = []
            times = np.full(K, np.nan)
            mstepsizes = np.full(K, np.nan)
            mrhats = np.full(K, np.nan)
            for k in range(K):

                (samples_k, max_sampling_time, mean_stepsize, max_rhat, _
                ) = stan_sample_subprocess(
                    model = stan_model_name,
                    pars = 'phi',
                    data = data_k[k],
                    seed = seeds[k],
                    chains = conf.chains,
                    iter = iters,
                    thin = 1,
                )
                times[k] = max_sampling_time
                mstepsizes[k] = mean_stepsize
                mrhats[k] = max_rhat
                samples.append(samples_k['phi'])

            # Moment estimates
            # TODO make more efficient similar as in Master.mix_phi()
            samp = np.concatenate(samples, axis=0)
            nsamp = samp.shape[0]
            samp.mean(axis=0, out=m_s_cons[i])
            samp -= m_s_cons[i]
            samp.T.dot(samp, out=S_s_cons[i])
            S_s_cons[i] /= nsamp - 1

            # diagnostics
            time_s_cons[i] = np.max(times)
            mstepsize_s_cons[i] = np.mean(mstepsizes)
            mrhat_s_cons[i] = np.max(mrhats)

            # dereference samples
            del samples, samp

        # Save results
        if conf.save_res:
            if not os.path.exists(RES_PATH):
                os.makedirs(RES_PATH)
            if conf.id:
                filename = 'res_c_{}_{}.npz'.format(model_name, conf.id)
            else:
                filename = 'res_c_{}.npz'.format(model_name)
            np.savez(
                os.path.join(RES_PATH, filename),
                conf = conf.__dict__,
                m_s_cons = m_s_cons,
                S_s_cons = S_s_cons,
                time_s_cons = time_s_cons,
                mstepsize_s_cons = mstepsize_s_cons,
                mrhat_s_cons = mrhat_s_cons,
            )
            print("Consensus MC results saved.")

        print("Done with consensus MC")

    # --------------------------------------------------------------------------
    #   Target sampling
    # --------------------------------------------------------------------------
    if conf.run_target or conf.run_all:

        print("Target approximation")

        seed = np.random.RandomState(seed=conf.seed_mcmc)

        data_target = dict(
            N = data.X.shape[0],
            D = data.X.shape[1],
            J = J,
            X = data.X,
            y = data.y,
            j_ind = data.j_ind+1,
            mu_phi = m0,
            Omega_phi = Q0.T    # Q0 transposed in order to get C-contiguous
        )
        # Load model
        stan_model = load_stan(os.path.join(MOD_PATH, model_name))

        # Sample and extract samples
        fit, time_target = stan_sample_time(
            stan_model,
            data = data_target,
            seed = seed,
            pars = 'phi',
            chains = conf.chains,
            iter = conf.target_siter,
            warmup = None,
            thin = 1,
        )
        samp = fit.extract(pars='phi')['phi']

        # Mean stepsize
        steps = [np.mean(p['stepsize__']) for p in fit.get_sampler_params()]
        print('    sampling time {}'.format(time_target))
        print('    mean stepsize: {:.4}'.format(np.mean(steps)))
        # Max Rhat (from all but last row in the last column)
        print('    max Rhat: {:.4}'.format(
            np.max(fit.summary()['summary'][:-1,-1])
        ))

        # Save samples
        if conf.save_target_samp:
            if not os.path.exists(RES_PATH):
                os.makedirs(RES_PATH)
            if conf.id:
                filename = 'target_samp_{}_{}.npz'.format(model_name, conf.id)
            else:
                filename = 'target_samp_{}.npz'.format(model_name)
            np.savez(
                os.path.join(RES_PATH, filename),
                conf = conf.__dict__,
                samp_target = samp
            )
            print("Target approximation samples saved.")

        # Moment estimates
        nsamp = samp.shape[0]
        m_target = samp.mean(axis=0)
        samp -= m_target
        S_target = samp.T.dot(samp)
        S_target /= nsamp - 1

        # Save results
        if conf.save_res:
            if not os.path.exists(RES_PATH):
                os.makedirs(RES_PATH)
            if conf.id:
                filename = 'target_{}_{}.npz'.format(model_name, conf.id)
            else:
                filename = 'target_{}.npz'.format(model_name)
            np.savez(
                os.path.join(RES_PATH, filename),
                conf = conf.__dict__,
                m_target = m_target,
                S_target = S_target,
                time_target = time_target,
            )
            print("Target results saved.")

        print("Done with target approximation")


def _create_pmaps(phiers, J, K, Ns):
    """Create the mappings for hierarhical parameters."""
    if K < 2:
        raise ValueError("K should be at least 2.")

    elif K < J:
        # ------ Many groups per site: combined groups ------
        pmaps = []
        for pi in range(len(phiers)):
            ih = phiers[pi]
            if ih is None:
                pmaps.append(None)
            else:
                pmap = []
                i = 0
                for k in range(K):
                    # Create indexings until the ih dimension, remaining
                    # dimension's slice(None) can be left out
                    if ih == 0:
                        pmap.append(slice(i, i+Ns[k]))
                    else:
                        pmap.append(
                            tuple(
                                slice(i, i+Ns[k])
                                if i2 == ih else slice(None)
                                for i2 in range(ih+1)
                            )
                        )
                    i += Ns[k]
                pmaps.append(pmap)

    elif K == J:
        # ------ One group per site ------
        pmaps = []
        for pi in range(len(phiers)):
            ih = phiers[pi]
            if ih is None:
                pmaps.append(None)
            elif ih == 0:
                # First dimensions can be mapped with one ndarray
                pmaps.append(np.arange(K))
            else:
                pmap = []
                for k in range(K):
                    # Create indexings until the ih dimension, remaining
                    # dimension's slice(None) can be left out
                    pmap.append(
                        tuple(
                            k if i2 == ih else slice(None)
                            for i2 in range(ih+1)
                        )
                    )
                pmaps.append(pmap)

    else:
        # ------ Multiple sites per group: split groups ------
        pmaps = []
        for pi in range(len(phiers)):
            ih = phiers[pi]
            if ih is None:
                pmaps.append(None)
            elif ih == 0:
                # First dimensions can be mapped with one ndarray
                pmap = np.empty(K, dtype=np.int32)
                i = 0
                for j in range(J):
                    for _ in range(Ns[j]):
                        pmap[i] = j
                        i += 1
                pmaps.append(pmap)
            else:
                pmap = []
                i = 0
                for j in range(J):
                    for _ in range(Ns[j]):
                        # Create indexings until the ih dimension, remaining
                        # dimension's slice(None) can be left out
                        pmap.append(
                            tuple(
                                j if i2 == ih else slice(None)
                                for i2 in range(ih+1)
                            )
                        )
                        i += 1
                pmaps.append(pmap)

    return pmaps


# ==============================================================================
# Command line argument parsing
# ==============================================================================

def _parse_bool(arg):
    up = str(arg).upper()
    if up == 'TRUE'[:len(up)] or up == '1':
       return True
    elif up == 'FALSE'[:len(up)] or up == '0':
       return False
    else:
       raise ValueError("Invalid boolean option")

def _parse_positive_int(arg):
    if arg.isalnum() and int(arg) > 0:
        return int(arg)
    else:
       raise ValueError("Invalid integer option")

def _parse_damp(arg):
    f = float(arg)
    if f <= 0.0  or f > 1.0:
        raise ValueError("Invalid damp option")
    return f

def _parse_nonnegative_int(arg):
    if arg.isalnum():
        return int(arg)
    else:
       raise ValueError("Invalid integer option")

CONF_HELP = dict(

    J                = 'number of hierarchical groups',
    D                = 'number of inputs',
    K                = 'number of sites',
    npg              = 'number of observations per group (constant or min max)',
    cor_input        = 'correlated input variable',

    run_all          = 'run all the methods',
    run_ep           = 'run the distributed EP method',
    run_full         = 'run the full model method',
    run_consensus    = 'run consensus MC method',
    run_target       = 'run target approximation',

    iter             = 'number of distributed EP iterations',
    siter            = 'Stan iterations in each major iteration',
    target_siter     = 'Stan iterations for the target approximation',
    chains           = 'number of chains used in stan sampling',

    damp             = 'damping factor constant',
    mix              = 'mix last iteration samples',
    prec_estim       = (
        'estimate method for tilted distribution precision '
        'matrix, currently available options are sample and olse '
        '(see epstan.method.Master)'
    ),

    seed_data        = 'seed for data simulation',
    seed_mcmc        = 'seed for sampling',

    id               = 'optional id appended to the end of the result files',
    save_true        = 'save true values',
    save_res         = 'save results',
    save_target_samp = 'save target approximation samples',

)

for conf in CONFS:
    if conf.startswith('mc_'):
        CONF_HELP[conf] = (
            CONF_HELP[conf] + ', default ({} {} {} {})'.format(*[
                CONF_DEFAULT[conf][k]
                for k in ['chains', 'iter', 'warmup', 'thin']
            ])
        )
    else:
        CONF_HELP[conf] = \
            CONF_HELP[conf] + ', default {}'.format(CONF_DEFAULT[conf])

CONF_CUSTOMS = dict(

    J                = dict(type=_parse_positive_int, metavar='P'),
    D                = dict(type=_parse_positive_int, metavar='P'),
    K                = dict(type=_parse_positive_int, metavar='P'),
    npg              = dict(nargs='+', type=_parse_positive_int, metavar='P'),
    cor_input        = dict(type=_parse_bool, metavar='B'),

    run_all          = dict(type=_parse_bool, metavar='B'),
    run_ep           = dict(type=_parse_bool, metavar='B'),
    run_full         = dict(type=_parse_bool, metavar='B'),
    run_consensus    = dict(type=_parse_bool, metavar='B'),
    run_target       = dict(type=_parse_bool, metavar='B'),

    iter             = dict(type=_parse_positive_int, metavar='P'),
    siter            = dict(type=_parse_positive_int, metavar='P'),
    target_siter     = dict(type=_parse_positive_int, metavar='P'),
    chains           = dict(type=_parse_positive_int, metavar='P'),

    damp             = dict(type=_parse_damp, metavar='F'),
    mix              = dict(type=_parse_bool, metavar='B'),
    prec_estim       = dict(metavar='S'),

    seed_data        = dict(type=_parse_nonnegative_int, metavar='N'),
    seed_mcmc        = dict(type=_parse_nonnegative_int, metavar='N'),

    id               = dict(metavar='S'),
    save_true        = dict(type=_parse_bool, metavar='B'),
    save_res         = dict(type=_parse_bool, metavar='B'),
    save_target_samp = dict(type=_parse_bool, metavar='B'),

)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description = __doc__.split('\n\n', 1)[0],
        epilog = "See module docstring for more detailed info.",
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('model_name', help = "name of the model")
    for opt in CONFS:
        parser.add_argument(
            '--'+opt,
            default = CONF_DEFAULT[opt],
            help = CONF_HELP[opt],
            **CONF_CUSTOMS[opt]
        )
    args = parser.parse_args()
    model_name = args.model_name
    args = vars(args)
    args.pop('model_name')

    # Process custom npg arg
    if isinstance(args['npg'], list):
        if len(args['npg']) == 1:
            args['npg'] = args['npg'][0]
        elif len(args['npg']) > 2:
            raise ValueError("Invalid arg `npg`, provide one or two elements")

    # Create configurations object
    conf = configurations(**args)

    # Run
    main(model_name, conf)
