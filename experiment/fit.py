"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

Execute with:
$ python fit.py [-h] [--J P] [--D P] [--K P] [--npg P [P ...]] [--iter N]
                [--cor_input B] [--damp F] [--prec_estim S]
                [--method {both,distributed,full,none}] [--id S] [--save_true B]
                [--save_res B] [--seed_data N] [--seed_mcmc N]
                [--mc_opt P P P P] [--mc_full_opt P P P P]
                model_name


positional arguments:
  model_name            name of the model

optional arguments:
  -h, --help            show this help message and exit
  --J P                 number of hierarchical groups, default 40
  --D P                 number of inputs, default 20
  --K P                 number of sites, default 25
  --npg P [P ...]       number of observations per group (constant or min
                        max), default [40, 60]
  --iter N              number of distributed EP iterations, default 6
  --cor_input B         correlated input variable, default False
  --damp F              damping factor constant, 1/K by default, default None
  --mix B               mix last iteration samples, default False
  --prec_estim S        estimate method for tilted distribution precision
                        matrix, currently available options are sample and
                        olse (see epstan.method.Master), default sample
  --method {both,distributed,full,none}
                        which models are fit, default both
  --id S                optional id appended to the end of the result files,
                        default None
  --save_full_samp B    save samples obtained from the full model, default False
  --save_true B         save true values, default True
  --save_res B          save results, default True
  --seed_data N         seed for data simulation, default 0
  --seed_mcmc N         seed for sampling, default 0
  --mc_opt P P P P      MCMC sampler opt for epstan (chains iter warmup thin),
                        default (4 400 200 1)
  --mc_full_opt P P P P
                        MCMC sampler opt for full (chains iter warmup thin),
                        default (4 1000 500 1)

Available models are defined in the folder models in the files
`<model_name>.py`, `<model_name>.stan` and `<model_name>_sg.stan`

Argument types
- N denotes a non-negative and P a positive integer argument.
- F denotes a float argument
- B denotes a boolean argument, which can be given as
  TRUE, T, 1 or FALSE, F, 0 (case insensitive).
- S denotes a string argument.

The results of full model are saved into file
    `res_f_<model_name>.npz`,
the results of distributed model are saved into file
    `res_d_<model_name>.npz`
and the true values are saved into the file
    `true_vals_<model_name>.npz`
into the folder results.

After running this skript for both full and distributed, the script plot_res.py
can be used to plot the results.

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import os, argparse
from timeit import default_timer as timer
import numpy as np

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
from epstan.util import load_stan, distribute_groups, suppress_stdout


CONFS = ['J','D', 'K', 'npg', 'iter', 'cor_input', 'damp', 'mix', 'prec_estim',
         'method', 'id', 'save_full_samp', 'save_true', 'save_res', 'seed_data',
         'seed_mcmc', 'mc_opt', 'mc_full_opt']

CONF_DEFAULT = dict(
    J              = 40,
    D              = 20,
    K              = 25,
    npg            = [40,60],
    iter           = 6,
    cor_input      = False,
    damp           = None,
    mix            = False,
    prec_estim     = 'sample',
    method         = 'both',
    id             = None,
    save_full_samp = False,
    save_true      = True,
    save_res       = True,
    seed_data      = 0,
    seed_mcmc      = 0,
    # MCMS sampler options for epstan method
    mc_opt = dict(
        chains     = 4,
        iter       = 400,
        warmup     = 200,
        thin       = 1,
    ),
    # MCMS sampler options for full method
    mc_full_opt = dict(
        chains     = 4,
        iter       = 1000,
        warmup     = 500,
        thin       = 1,
    )
)

# Temp fix for the RandomState seed problem with pystan in 32bit Python.
# Detect automatically if in 32bit mode
TMP_FIX_32BIT = os.sys.maxsize <= 2**32


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
        data = model.simulate_data(Sigma_x='rand', seed=conf.seed_data)
    else:
        data = model.simulate_data(seed=conf.seed_data)

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

    # ------------------------------------------------------
    #     Fit distributed model
    # ------------------------------------------------------
    if conf.method == 'both' or conf.method == 'distributed' or ret_master:

        print("Distributed model {} ...".format(model_name))

        # Custom sinusoidal damping factor function
        if conf.damp is None:
            df0_start = 0.01
            df0_end = 0.25
            df0 = lambda i: (
                df0_start + (df0_end - df0_start) * 0.5 * (1 + np.sin(
                np.pi * (max(0,min(i-2,conf.iter-2))/(conf.iter-2) - 0.5)))
            )
        else:
            df0 = conf.damp

        # Options for the ep-algorithm see documentation of epstan.method.Master
        epstan_options = dict(
            prior = prior,
            seed = conf.seed_mcmc,
            prec_estim = conf.prec_estim,
            df0 = df0,
            init_site = init_site,
            **conf.mc_opt
        )
        # Temp fix for the RandomState seed problem with pystan in 32bit Python
        epstan_options['tmp_fix_32bit'] = TMP_FIX_32BIT

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
            Nk, Nk_j, _ = distribute_groups(J, K, data.Nj)
            # Create the Master instance
            epstan_master = Master(
                os.path.join(MOD_PATH, model_name+'_sg'),
                data.X,
                data.y,
                site_sizes=Nk,
                **epstan_options
            )
            # Construct the map: which site contribute to which parameter
            pmaps = _create_pmaps(phiers, J, K, Nk_j)

        else:
            raise ValueError("K cant be greater than number of samples")

        if ret_master:
            print("Returning epstan.Master")
            return epstan_master

        # Run the algorithm for `EP_ITER` iterations
        print("Run distributed EP algorithm for {} iterations." \
              .format(conf.iter))
        if conf.mix:
            m_phi_i, cov_phi_i, info = epstan_master.run(
                conf.iter, save_last_param=pnames)
        else:
            m_phi_i, cov_phi_i, info = epstan_master.run(conf.iter)
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
                    conf      = conf.__dict__,
                    m_phi_i   = m_phi_i,
                    cov_phi_i = cov_phi_i,
                    last_iter = epstan_master.iter
                )
                print("Uncomplete distributed model results saved.")
            raise RuntimeError('epstan algorithm failed with error code: {}'
                               .format(info))

        if conf.mix:
            print("Form the final approximation "
                  "by mixing the last samples from all the sites.")
            cov_phi, m_phi = epstan_master.mix_phi()

            # Get mean and var of inferred variables
            pms, pvars = epstan_master.mix_pred(pnames, pmaps, pshapes)
            # Construct a dict of from these results
            presults = {}
            for i in range(len(pnames)):
                pname = pnames[i]
                presults['m_'+pname] = pms[i]
                presults['var_'+pname] = pvars[i]

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
                    conf      = conf.__dict__,
                    m_phi_i   = m_phi_i,
                    cov_phi_i = cov_phi_i,
                    m_phi     = m_phi,
                    cov_phi   = cov_phi,
                    **presults
                )
            else:
                np.savez(
                    os.path.join(RES_PATH, filename),
                    conf      = conf.__dict__,
                    m_phi_i   = m_phi_i,
                    cov_phi_i = cov_phi_i,
                )
            print("Distributed model results saved.")

        # Release master object
        del epstan_master

    # ------------------------------------------------------
    #     Fit full model
    # ------------------------------------------------------
    if conf.method == 'both' or conf.method == 'full':

        print("Full model {} ...".format(model_name))

        seed = np.random.RandomState(seed=conf.seed_mcmc)
        # Temp fix for the RandomState seed problem with pystan in 32bit Python
        seed = seed.randint(2**31-1) if TMP_FIX_32BIT else seed

        data = dict(
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
        time_full = timer()
        fit = stan_model.sampling(
            data = data,
            seed = seed,
            pars = 'phi',
            **conf.mc_full_opt
        )
        time_full = (timer() - time_full)
        samp = fit.extract(pars='phi')['phi']

        # Mean stepsize
        steps = [np.mean(p['stepsize__']) for p in fit.get_sampler_params()]
        print('    sampling time {}'.format(time_full))
        print('    mean stepsize: {:.4}'.format(np.mean(steps)))
        # Max Rhat (from all but last row in the last column)
        print('    max Rhat: {:.4}'.format(
            np.max(fit.summary()['summary'][:-1,-1])
        ))

        # Save samples
        if conf.save_full_samp:
            if not os.path.exists(RES_PATH):
                os.makedirs(RES_PATH)
            if conf.id:
                filename = 'full_samp_{}_{}.npz'.format(model_name, conf.id)
            else:
                filename = 'full_samp_{}.npz'.format(model_name)
            np.savez(
                os.path.join(RES_PATH, filename),
                conf = conf.__dict__,
                samp_phi = samp
            )
            print("Full model samples saved.")

        # Moment estimates
        nsamp = samp.shape[0]
        m_phi_full = samp.mean(axis=0)
        samp -= m_phi_full
        cov_phi_full = samp.T.dot(samp)
        cov_phi_full /= nsamp -1

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
                conf         = conf.__dict__,
                m_phi_full   = m_phi_full,
                cov_phi_full = cov_phi_full,
            )
            print("Full model results saved.")


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
    J              = 'number of hierarchical groups',
    D              = 'number of inputs',
    K              = 'number of sites',
    npg            = 'number of observations per group (constant or min max)',
    iter           = 'number of distributed EP iterations',
    cor_input      = 'correlated input variable',
    damp           = 'damping factor constant',
    mix            = 'mix last iteration samples',
    prec_estim     = ('estimate method for tilted distribution precision '
                      'matrix, currently available options are sample and olse '
                      '(see epstan.method.Master)'),
    method         = 'which models are fit',
    id             = 'optional id appended to the end of the result files',
    save_full_samp = 'save samples obtained from the full model',
    save_true      = 'save true values',
    save_res       = 'save results',
    seed_data      = 'seed for data simulation',
    seed_mcmc      = 'seed for sampling',
    mc_opt         = 'MCMC sampler opt for epstan (chains iter warmup thin)',
    mc_full_opt    = 'MCMC sampler opt for full (chains iter warmup thin)',
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
    J              = dict(type=_parse_positive_int, metavar='P'),
    D              = dict(type=_parse_positive_int, metavar='P'),
    K              = dict(type=_parse_positive_int, metavar='P'),
    npg            = dict(nargs='+', type=_parse_positive_int, metavar='P'),
    iter           = dict(type=_parse_nonnegative_int, metavar='N'),
    cor_input      = dict(type=_parse_bool, metavar='B'),
    damp           = dict(type=_parse_damp, metavar='F'),
    mix            = dict(type=_parse_bool, metavar='B'),
    prec_estim     = dict(metavar='S'),
    method         = dict(choices=['both', 'distributed', 'full', 'none']),
    id             = dict(metavar='S'),
    save_full_samp = dict(type=_parse_bool, metavar='B'),
    save_true      = dict(type=_parse_bool, metavar='B'),
    save_res       = dict(type=_parse_bool, metavar='B'),
    seed_data      = dict(type=_parse_nonnegative_int, metavar='N'),
    seed_mcmc      = dict(type=_parse_nonnegative_int, metavar='N'),
    mc_opt         = dict(nargs=4, type=_parse_positive_int, metavar='P'),
    mc_full_opt    = dict(nargs=4, type=_parse_positive_int, metavar='P'),
)

if __name__ == '__main__':

    # Process help string
    descr_ind = __doc__.find('\n\n')
    epilog_ind = __doc__.find('optional arguments:\n')
    epilog_ind = __doc__.find('\n\n', epilog_ind)
    if descr_ind == -1 or epilog_ind == -1:
        description = None
        epilog = __doc__
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    else:
        description = __doc__[:descr_ind]
        epilog = __doc__[epilog_ind+2:]
        formatter_class = argparse.RawDescriptionHelpFormatter

    # Parse arguments
    parser = argparse.ArgumentParser(
        description = description,
        epilog = epilog,
        formatter_class = formatter_class
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

    # Process customs
    if not isinstance(args['mc_opt'], dict):
        args['mc_opt'] = dict(
            chains = args['mc_opt'][0],
            iter   = args['mc_opt'][1],
            warmup = args['mc_opt'][2],
            thin   = args['mc_opt'][3],
        )
    if not isinstance(args['mc_full_opt'], dict):
        args['mc_full_opt'] = dict(
            chains = args['mc_full_opt'][0],
            iter   = args['mc_full_opt'][1],
            warmup = args['mc_full_opt'][2],
            thin   = args['mc_full_opt'][3],
        )
    if isinstance(args['npg'], list):
        if len(args['npg']) == 1:
            args['npg'] = args['npg'][0]
        elif len(args['npg']) > 2:
            raise ValueError("Invalid arg `npg`, provide one or two elements")

    # Create configurations object
    conf = configurations(**args)

    # Run
    main(model_name, conf)
