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
git clone git@github.com/CAHLR/xBKT.git
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

Similarly, if working in OS X, you can download the latest stable version of Eigen 
from the site above. This program has run successfully with `Eigen 3.2.5`.
First move the file to /usr/local/include, then unzip and create simplified link to Eigen. 
These commands can be used below:


    mv <path to file>/3.1.3.tar.gz /usr/local/include/3.1.3.tar.gz
    tar -xvf 3.1.3.tar.gz
    ln -s <name of unzipped file>/Eigen ./Eigen
    rm 3.1.3.tar.gz


## Compiling ##

Run `make` in the root directory of the xBKT project folder. If this step runs successfully, you should see a MEX file generated for each of the .cpp files. 

## Potential Errors When Running Makefile##

Before running `make`, check `Makefile` in xBKT. Be sure that the `MATLABPATH` matches your matlab version and `EIGENPATH` matches your Eigen filepath. For example, if you're working with Matlab 2015 in OS X, you may need to update `Makefile` with the new name of your `Applications` from

```
    ifeq ($(UNAME),Darwin)
        MATLABPATH=/Applications/MATLAB_R2013a.app
    endif
```

to something like


```
    ifeq ($(UNAME),Darwin)
        MATLABPATH=/Applications/MATLAB_R2015b.app
    endif
```    


You may also see the following error while running `make`
```
    make: g++-4.9: No such file or directory
```
If you see this error, you need to download gcc49. This can be downloaded with [brew](http://brew.sh/). 

These steps would allow you to set up gcc49. Run the following commands
```
    brew install --enable-cxx gcc49
    brew install mpfr
    brew install gmp
    brew install libmpc
```
You may also need to change `CXX=g++-4.9` to `CXX=g++-5` in `Makefile`, depending on the version you set up. 


