Created by Zachary A. Pardos (zp@berkeley.edu) and Matthew J. Johnson (mattjj@csail.mit.edu)
Computational Approaches to Human Learning Research (CAHL) Lab @ UC Berkeley
# Running #

You can work in the repository root directory or add it to your path with
`addpath` (no need to use `genpath`, since everything is organized with
namespaces).

See the file `+test/hand_specified_model.m` for a fairly complete example,
which you can run with `test.hand_specified_model`.

If you get an error about missing the function `E_step`, the compiling step
(see "Installation and setup") didn't work.

Here's a simplified version:

```matlab
%% generate synthetic model and data
% the model will have 4 question subparts
num_subparts = 4;
truemodel = generate.random_model(num_subparts);

% generate 3 observation sequences with varying lengths
data = generate.synthetic_data(truemodel,[200,150,500]);

%% fit models, starting with random initializations
best_likelihood = -inf;
for i=1:25
    [fitmodel, log_likelihoods] = fit.EM_fit(generate.random_model(num_subparts),data);
    if (log_likelihoods(end) > best_likelihood)
        best_likelihood = log_likelihoods(end);
        best_model = fitmodel;
    end
end

%% compare the fit model to the true model
disp('these two should look similar');
truemodel.A
best_model.A
```

# Installation and setup #

## Cloning the repository ##

```
git clone git@github.com:mattjj/edX-hmm.git
```

## Installing Eigen ##

Get Eigen from http://eigen.tuxfamily.org/index.php?title=Main_Page and unzip
it somewhere (anywhere will work, but it affects the mex command below). On a
\*nix machine, these commands should put Eigen in /usr/local/include:

    cd /usr/local/include
    wget --no-check-certificate http://bitbucket.org/eigen/eigen/get/3.1.3.tar.gz
    tar -xzvf 3.1.3.tar.gz
    ln -s eigen-eigen-2249f9c22fe8/Eigen ./Eigen
    rm 3.1.3.tar.gz

## Compiling ##

Just run `make` in the top-level directory.

