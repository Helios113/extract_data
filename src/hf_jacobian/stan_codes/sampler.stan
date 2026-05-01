data {
    int<lower=1> d;
    int<lower=d+1> D;
    matrix[D, d+1] frame;
    real<lower=0> R;          // Bounding radius for the base domain
    vector[d] lambdas;        // Quadratic coefficients (same sign = elliptic, mixed = hyperbolic)
    real<lower=0> noise_std;
}
parameters {
    // Parameterize the base domain U as a d-ball using spherical coordinates
    real<lower=0, upper=R> r;
    unit_vector[d] w;
}
model {
    // 1. Jacobian for spherical coordinates in R^d
    target += (d - 1) * log(r);
    
    // 2. Base domain coordinates
    vector[d] x = r * w;
    
    // 3. Monge patch surface area element correction
    // f(x) = sum_i(lambda_i * x_i^2)
    // grad_f(x)_i = 2 * lambda_i * x_i
    vector[d] grad_f = 2 * lambdas .* x;
    
    // target += log(sqrt(1 + ||grad f(x)||^2))
    target += 0.5 * log(1 + dot_self(grad_f));
}
generated quantities {
    vector[d] x = r * w;
    real z = sum(lambdas .* square(x));
    
    vector[d+1] local_pt;
    local_pt[1:d] = x;
    local_pt[d+1] = z;
    
    vector[D] pt_clean = frame * local_pt;
    vector[D] pt;
    if (noise_std > 0) {
        for (i in 1:D) {
            pt[i] = pt_clean[i] + normal_rng(0, noise_std);
        }
    } else {
        pt = pt_clean;
    }
}