
data {
    int<lower=0> N;
    int<lower=0> K;
    int<lower=0> J;
    matrix[N,K] X;
    int<lower=0,upper=1> y[N];
    int<lower=1,upper=J> jj[N];
    vector[K+1] mu_prior;
    cov_matrix[K+1] Sigma_prior;
}
parameters {
    vector[K+1] phi;
    vector[J] eta;
}
transformed parameters {
    vector[J] alpha;
    real<lower=0> sigma_a;
    sigma_a <- exp(phi[1]);
    alpha <- eta * sigma_a;
}
model {
    vector[N] f;
    vector[K] beta;
    eta ~ normal(0, 1);
    phi ~ multi_normal(mu_prior, Sigma_prior);
    beta <- tail(phi, K);
    for (n in 1:N){
        f[n] <- alpha[jj[n]] + X[n]*beta;
    }
    y ~ bernoulli_logit(f);
}

