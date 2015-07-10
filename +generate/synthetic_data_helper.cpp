#include <stdint.h>
#include <Eigen/Core>
#include "mex.h"

using namespace Eigen;
using namespace std;

// TODO openmp version

// NOTE: mxSetProperty and possibly mxGetProperty make copies, even with
// classdef < handle! lame! I believe mxGetField does NOT make a copy, though I
// haven't tested it recently

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    /* SETUP */
    if (nrhs != 4) { mexErrMsgTxt("wrong number of arguments\n"); }

    //// pull out inputs

    // parameters struct
    if (!mxGetField(prhs[0],0,"learns")) { mexErrMsgTxt("model missing 'learns' field\n"); }
    double *learns = mxGetPr(mxGetField(prhs[0],0,"learns"));
    int num_resources = max(mxGetM(mxGetField(prhs[0],0,"learns")),
            mxGetN(mxGetField(prhs[0],0,"learns")));

    if (!mxGetField(prhs[0],0,"forgets")) { mexErrMsgTxt("model missing 'forgets' field\n"); }
    double *forgets = mxGetPr(mxGetField(prhs[0],0,"forgets"));

    if (!mxGetField(prhs[0],0,"guesses")) { mexErrMsgTxt("model missing 'guesses' field\n"); }
    double *guess = mxGetPr(mxGetField(prhs[0],0,"guesses"));

    if (!mxGetField(prhs[0],0,"slips")) { mexErrMsgTxt("model missing 'slips' field\n"); }
    double *slip = mxGetPr(mxGetField(prhs[0],0,"slips"));

    int num_subparts = max(mxGetM(mxGetField(prhs[0],0,"slips")),
            mxGetN(mxGetField(prhs[0],0,"slips")));

    if (!mxGetField(prhs[0],0,"prior")) { mexErrMsgTxt("model missing 'prior' field\n"); }
    double prior = mxGetScalar(mxGetField(prhs[0],0,"prior"));

    Vector2d initial_distn;
    initial_distn << 1-prior, prior;

    MatrixXd As(2,2*num_resources);
    for (int n=0; n<num_resources; n++) {
        As.col(2*n) << 1-learns[n], learns[n];
        As.col(2*n+1) << forgets[n], 1-forgets[n];
    }

    int32_t *starts = (int32_t *) mxGetData(prhs[1]);
    int32_t *lengths = (int32_t *) mxGetData(prhs[2]);
    int16_t *allresources = (int16_t *) mxGetData(prhs[3]);

    int num_sequences = max(mxGetM(prhs[1]),mxGetN(prhs[1]));

    int bigT = 0;
    for (int k=0; k<num_sequences; k++) {
        bigT += (int) lengths[k];
    }

    //// outputs

    if (nlhs != 2) { mexErrMsgTxt("needs two outputs\n"); }

    // NOTE: these are doubles just to stick with the convention we had going
    // with the Matlab code, but they really should be some integer type. Some
    // implicit casting with the indexing below.
    plhs[0] = mxCreateNumericMatrix(1,bigT,mxINT8_CLASS,mxREAL);
    plhs[1] =  mxCreateNumericMatrix(num_subparts,bigT,mxINT8_CLASS,mxREAL);
    int8_t *all_stateseqs = (int8_t *) mxGetData(plhs[0]);
    int8_t *all_data = (int8_t *) mxGetData(plhs[1]);

    /* COMPUTATION */

    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        // NOTE: -1 because Matlab indexing starts at 1
        int32_t sequence_start = starts[sequence_index] - 1;
        int32_t T = lengths[sequence_index];

        int8_t *data = all_data + num_subparts*sequence_start;
        int16_t *resources = allresources + sequence_start;
        int8_t *stateseq = all_stateseqs + sequence_start;

        Vector2d nextstate_distr = initial_distn;
        for (int t=0; t<T; t++) {
            stateseq[t] = nextstate_distr(0) < ((double) rand()) / ((double) RAND_MAX);
            for (int n=0; n<num_subparts; n++) {
                data[n+num_subparts*t] = (stateseq[t] ? slip[n] : (1-guess[n]))
                    < ((double) rand()) / ((double) RAND_MAX);
            }
            nextstate_distr = As.col(2*(resources[t]-1)+stateseq[t]);
        }
    }
}

