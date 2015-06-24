"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869). This experiment uses importance sampling instead of Stan.

Execute with:
$ python fit.py [-h] [--J P] [--D P] [--K P] [--npg P [P ...]] [--iter N]
                [--prec_estim S] [--method {both,distributed,full,none}]
                [--id S] [--save_true B] [--save_res B] [--seed_data N]
                [--seed_inf N] [--mc_opt P P P P] [--mc_full_opt P P P P]
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
  --method {both,distributed,full,none}
                        which models are fit, default both
  --id S                optional id appended to the end of the result files,
                        default None
  --save_true B         save true values, default True
  --save_res B          save results, default True
  --seed_data N         seed for data simulation, default 0
  --seed_inf N          seed for sampling, default 0
  --nsamp P             Number of samples used in the tilted distribution
                        inference, default 1000
  --mc_full_opt P P P P
                        MCMC sampler opt for full (chains iter warmup thin),
                        default (4 1000 500 2)

Available models are defined in the folder models in the files 
`<model_name>.py`, `<model_name>.stan` and `<model_name>_sg.stan`

Argument types
- N denotes a non-negative and P a positive integer argument.
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

from __future__ import division
import os, argparse
import numpy as np

# Add parent dir to sys.path if not present already. This is only done because
# of easy importing of the package dep. Adding the parent directory into the
# PYTHONPATH works as well.
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(CUR_PATH, os.pardir))
RES_PATH = os.path.join(CUR_PATH, 'results')
MOD_PATH = os.path.join(CUR_PATH, 'models')
# Double check that the package is in the parent directory
if os.path.exists(os.path.join(PARENT_PATH, 'dep')):
    if PARENT_PATH not in os.sys.path:
        os.sys.path.insert(0, PARENT_PATH)

from dep.is_tilted import Master
from dep.util import load_stan, distribute_groups, suppress_stdout


CONFS = ['J','D', 'K', 'npg', 'iter', 'cor_input', 'method', 'id',
         'save_true', 'save_res', 'seed_data', 'seed_inf', 'nsamp',
         'mc_full_opt']

CONF_DEFAULT = dict(
    J           = 40,
    D           = 20,
    K           = 25,
    npg         = [40,60],
    iter        = 6,
    cor_input   = False,
    method      = 'both',
    id          = None,
    save_true   = True,
    save_res    = True,
    seed_data   = 0,
    seed_inf    = 0,
    nsamp       = 1000,
    # MCMC sampler options for full method
    mc_full_opt = dict(
        chains  = 4,
        iter    = 1000,
        warmup  = 500,
        thin    = 2,
    )
)

# Temp fix for the RandomState seed problem with pystan in 32bit Python.
# Detect automatically if in 32bit mode
TMP_FIX_32BIT = os.sys.maxsize <= 2**32


class configurations(object):
    """Configuration container for the function main."""
    def __init__(self, **kwargs):
        # Set given options
        for k, v in kwargs.iteritems():
            if k not in CONF_DEFAULT:
                raise ValueError("Invalid option `{}`".format(k))
            setattr(self, k, v)
        # Set missing options to defaults
        for k, v in CONF_DEFAULT.iteritems():
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
    
    Arg. `ret_master` can be used to prematurely exit and return the dep.Master
    object, which is useful for debuging.
    
    """
    
    # Ensure that the configurations class is used
    if not isinstance(conf, configurations):
        raise ValueError("Invalid arg. `conf`, use class fit.configurations")
    
    print "Configurations:"
    print '    ' + str(conf).replace('\n', '\n    ')
    
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
        print "True values saved into results"
    
    # ------------------------------------------------------
    #     Fit distributed model
    # ------------------------------------------------------
    if conf.method == 'both' or conf.method == 'distributed' or ret_master:
        
        print "Distributed model {} ...".format(model_name)
        
        # Options for the ep-algorithm see documentation of dep.is_tilted.Master
        dep_options = dict(
            prior      = prior,
            seed       = conf.seed_inf,
            nsamp      = conf.nsamp
        )
        # Create the Master instance
        liks = model.get_liks(K, data)
        dep_master = Master(liks, **dep_options)
        
        if ret_master:
            print "Returning dep.Master"
            return dep_master
        
        # Run the algorithm for `EP_ITER` iterations
        print "Run distributed EP algorithm for {} iterations." \
              .format(conf.iter)
        m_phi_i, cov_phi_i, info = dep_master.run(conf.iter)
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
                    last_iter = dep_master.iter
                )
                print "Uncomplete distributed model results saved."
            raise RuntimeError('Dep algorithm failed with error code: {}'
                               .format(info))
        print "Form the final approximation " \
              "by mixing the samples from all the sites."
        cov_phi, m_phi = dep_master.mix_phi()
        
        # Save results
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
                m_phi     = m_phi,
                cov_phi   = cov_phi,
            )
            print "Distributed model results saved."
        
        # Release master object
        del dep_master
    
    # ------------------------------------------------------
    #     Fit full model
    # ------------------------------------------------------
    if conf.method == 'both' or conf.method == 'full':
        
        print "Full model {} ...".format(model_name)
        
        seed = np.random.RandomState(seed=conf.seed_inf)
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
        # Load model if not loaded already
        if not 'stan_model' in locals():
            stan_model = load_stan(os.path.join(MOD_PATH, model_name))
        
        # Sample and extract parameters
        with suppress_stdout():
            fit = stan_model.sampling(
                data = data,
                seed = seed,
                **conf.mc_full_opt
            )
        samp = fit.extract(pars='phi')['phi']
        nsamp = samp.shape[0]
        m_phi_full = samp.mean(axis=0)
        samp -= m_phi_full
        cov_phi_full = samp.T.dot(samp)
        cov_phi_full /= nsamp -1
        
        # Mean stepsize
        steps = [np.mean(p['stepsize__'])
                 for p in fit.get_sampler_params()]
        print '    mean stepsize: {:.4}'.format(np.mean(steps))
        # Max Rhat (from all but last row in the last column)
        print '    max Rhat: {:.4}'.format(
            np.max(fit.summary()['summary'][:-1,-1])
        )
        
        # Get mean and var of inferred variables
        presults = {}
        for i in xrange(len(pnames)):
            pname = pnames[i]
            samp = fit.extract(pname)[pname]
            presults['m_'+pname+'_full'] = np.mean(samp, axis=0)
            presults['var_'+pname+'_full'] = np.var(samp, axis=0, ddof=1)
        
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
                **presults
            )
            print "Full model results saved."


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

def _parse_nonnegative_int(arg):
    if arg.isalnum():
        return int(arg)
    else:
       raise ValueError("Invalid integer option")

CONF_HELP = dict(
    J           = 'number of hierarchical groups',
    D           = 'number of inputs',
    K           = 'number of sites',
    npg         = 'number of observations per group (constant or min max)',
    iter        = 'number of distributed EP iterations',
    cor_input   = 'correlated input variable',
    method      = 'which models are fit',
    id          = 'optional id appended to the end of the result files',
    save_true   = 'save true values',
    save_res    = 'save results',
    seed_data   = 'seed for data simulation',
    seed_inf    = 'seed for sampling',
    nsamp       = 'Number of samples used in the tilted distribution inference',
    mc_full_opt = 'MCMC sampler opt for full (chains iter warmup thin)',
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
    J           = dict(type=_parse_positive_int, metavar='P'),
    D           = dict(type=_parse_positive_int, metavar='P'),
    K           = dict(type=_parse_positive_int, metavar='P'),
    npg         = dict(nargs='+', type=_parse_positive_int, metavar='P'),
    iter        = dict(type=_parse_nonnegative_int, metavar='N'),
    cor_input   = dict(type=_parse_bool, metavar='B'),
    method      = dict(choices=['both', 'distributed', 'full', 'none']),
    id          = dict(metavar='S'),
    save_true   = dict(type=_parse_bool, metavar='B'),
    save_res    = dict(type=_parse_bool, metavar='B'),
    seed_data   = dict(type=_parse_nonnegative_int, metavar='N'),
    seed_inf    = dict(type=_parse_nonnegative_int, metavar='N'),
    nsamp       = dict(type=_parse_positive_int, metavar='P'),
    mc_full_opt = dict(nargs=4, type=_parse_positive_int, metavar='P'),
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



