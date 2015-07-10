#include <Eigen/Core>
#include "mex.h"

using namespace Eigen;
using namespace std;

// TODO openmp version
// TODO roll counting in with second message passing
// TODO scaling instead of logs?

// NOTE: mxSetProperty and possibly mxGetProperty make copies, even with
// classdef < handle! lame! I believe mxGetField does NOT make a copy, though I
// haven't tested it recently

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{

    /* SETUP */
    if (nrhs != 5) { mexErrMsgTxt("wrong number of arguments\n"); }

    //// pull out inputs

    // data
    if (!mxGetField(prhs[0],0,"data")) { mexErrMsgTxt("data missing 'data' field\n"); }
    double *alldata = mxGetPr(mxGetField(prhs[0],0,"data"));
    int bigT = mxGetN(mxGetField(prhs[0],0,"data")); // total length of the data
    int num_subparts = mxGetM(mxGetField(prhs[0],0,"data")); // number of sub-parts

    if (!mxGetField(prhs[0],0,"resources")) { mexErrMsgTxt("data missing 'resources' field\n"); }
    double *allresources = mxGetPr(mxGetField(prhs[0],0,"resources"));

    if (!mxGetField(prhs[0],0,"starts")) { mexErrMsgTxt("data missing 'starts' field\n"); }
    double *starts = mxGetPr(mxGetField(prhs[0],0,"starts"));
    int num_sequences = max(mxGetM(mxGetField(prhs[0],0,"starts")),
            mxGetM(mxGetField(prhs[0],0,"starts")));

    if (!mxGetField(prhs[0],0,"lengths")) { mexErrMsgTxt("data missing 'lengths' field\n"); }
    double *lengths = mxGetPr(mxGetField(prhs[0],0,"lengths"));

    // parameters struct
    if (!mxGetField(prhs[1],0,"learns")) { mexErrMsgTxt("model missing 'learns' field\n"); }
    double *learns = mxGetPr(mxGetField(prhs[1],0,"learns"));
    int num_resources = max(mxGetM(mxGetField(prhs[1],0,"learns")),
            mxGetN(mxGetField(prhs[1],0,"learns")));

    if (!mxGetField(prhs[1],0,"forgets")) { mexErrMsgTxt("model missing 'forgets' field\n"); }
    double *forgets = mxGetPr(mxGetField(prhs[1],0,"forgets"));

    if (!mxGetField(prhs[1],0,"guesses")) { mexErrMsgTxt("model missing 'guesses' field\n"); }
    double *guess = mxGetPr(mxGetField(prhs[1],0,"guesses"));

    if (!mxGetField(prhs[1],0,"slips")) { mexErrMsgTxt("model missing 'slips' field\n"); }
    double *slip = mxGetPr(mxGetField(prhs[1],0,"slips"));

    if (!mxGetField(prhs[1],0,"prior")) { mexErrMsgTxt("model missing 'prior' field\n"); }
    double prior = mxGetScalar(mxGetField(prhs[1],0,"prior"));

    Array2d initial_distn;
    initial_distn << 1-prior, prior;

    MatrixXd As(2,2*num_resources);
    for (int n=0; n<num_resources; n++) {
        As.col(2*n) << 1-learns[n], learns[n];
        As.col(2*n+1) << forgets[n], 1-forgets[n];
    }
    ArrayXXd Als(2,2*num_resources);
    Als = As.array().log();

    Array2Xd Bn(2,2*num_subparts);
    for (int n=0; n<num_subparts; n++) {
        Bn.col(2*n) << log(1-guess[n]), log(slip[n]); // incorrect
        Bn.col(2*n+1) << log(guess[n]), log(1-slip[n]); // correct
    }

    //// outputs

    // rhs outputs
    Map<ArrayXXd,Aligned> all_trans_softcounts(mxGetPr(prhs[2]),2,2*num_resources);
    all_trans_softcounts.setZero();
    Map<Array2Xd,Aligned> all_emission_softcounts(mxGetPr(prhs[3]),2,2*num_subparts);
    all_emission_softcounts.setZero();
    Map<Array2d,Aligned> all_initial_softcounts(mxGetPr(prhs[4]));
    all_initial_softcounts.setZero();

    // lhs outputs
    Map<Array2Xd,Aligned> alphal_out(NULL,2,bigT);
    Map<Array2Xd,Aligned> betal_out(NULL,2,bigT);
    double *total_loglike = NULL;
    switch (nlhs) {
        case 3:
            plhs[2] = mxCreateDoubleMatrix(2,bigT,mxREAL);
            new (&betal_out) Map<Array2Xd,Aligned>(mxGetPr(plhs[2]),2,bigT);
        case 2:
            plhs[1] = mxCreateDoubleMatrix(2,bigT,mxREAL);
            new (&alphal_out) Map<Array2Xd,Aligned>(mxGetPr(plhs[1]),2,bigT);
        case 1:
            plhs[0] = mxCreateDoubleScalar(0.);
            total_loglike = mxGetPr(plhs[0]);
    }

    /* COMPUTATION */

    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        // NOTE: -1 because Matlab indexing starts at 1
        int sequence_start = ((int) starts[sequence_index]) - 1;
        int T = (int) lengths[sequence_index];

        Map<ArrayXXd> data(alldata + num_subparts*sequence_start,num_subparts,T);
        double *resources = allresources + sequence_start;

        //// likelihoods
        Array2Xd likelihoods(2,T);
        likelihoods.setZero();
        for (int t=0; t<T; t++) {
            for (int n=0; n<num_subparts; n++) {
                if (data(n,t) != 0) {
                    likelihoods.col(t) += Bn.col(2*n + (data(n,t) == 2));
                }
            }
        }

        //// forward messages
        Array2Xd alphal(2,T);
        alphal.col(0) = initial_distn.log() + likelihoods.col(0);
        for (int t=0; t<T-1; t++) {
            double cmax = alphal.col(t).maxCoeff();
            alphal.col(t+1) = (As.block(0,2*(resources[t]-1),2,2) // NOTE matlab indexing
                    * (alphal.col(t) - cmax).exp().matrix()).array().log()
                    + cmax + likelihoods.col(t+1);
        }

        //// backward messages
        Array2Xd betal(2,T);
        betal.col(T-1).setZero();
        for (int t=T-2; t>=0; t--) {
            Array2d thesum = likelihoods.col(t+1) + betal.col(t+1);
            double cmax = thesum.maxCoeff();
            betal.col(t) = (As.block(0,2*(resources[t]-1),2,2).transpose() // NOTE matlab indexing
                    * (thesum - cmax).exp().matrix()).array().log() + cmax;
        }

        //// statistic counting
        // TODO could roll this into second message sweep
        Array2d temp = alphal.col(0) + betal.col(0);
        double cmax = temp.maxCoeff();
        double loglike = log((temp - cmax).exp().sum()) + cmax;

        ArrayXXd trans_softcounts(2,2*num_resources);
        trans_softcounts.setZero();
        Array2Xd emission_softcounts(2,2*num_subparts);
        emission_softcounts.setZero();

        for (int t=0; t<T-1; t++) {
            Array22d pair; // cols = t, rows=t+1
            pair.rowwise() = alphal.col(t).transpose();
            pair.colwise() += (betal.col(t+1) + likelihoods.col(t+1));
            pair += Als.block(0,2*(resources[t]-1),2,2); // NOTE matlab indexing
            pair -= loglike;
            pair = pair.exp();

            trans_softcounts.block(0,2*(resources[t]-1),2,2) += pair; // NOTE matlab indexing

            Array2d single = pair.colwise().sum();
            for (int n=0; n<num_subparts; n++) {
                if (data(n,t) != 0) {
                    emission_softcounts.col(2*n + (data(n,t) == 2)) += single;
                }
            }
        }
        // NOTE: last iteration needed for emission softcounts
        int t=T-1;
        Array2d single = alphal.col(t);
        single -= loglike;
        single = single.exp();
        for (int n=0; n<num_subparts; n++) {
            if (data(n,t) != 0) { emission_softcounts.col(2*n + (data(n,t) == 2)) += single;
            }
        }

        Array2d initial_softcounts = betal.col(0) + alphal.col(0);
        initial_softcounts -= loglike;
        initial_softcounts = initial_softcounts.exp();

        // NOTE: these will need to change to base types for openmp
        all_trans_softcounts += trans_softcounts;
        all_emission_softcounts += emission_softcounts;
        all_initial_softcounts += initial_softcounts;

        switch (nlhs) {
            case 3:
                betal_out.block(0,sequence_start,2,T) = betal;
            case 2:
                alphal_out.block(0,sequence_start,2,T) = alphal;
            case 1:
                *total_loglike += loglike;
        }

    }
}

