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
    if (nrhs != 3) { mexErrMsgTxt("wrong number of arguments\n"); }

    //// pull out inputs

    // data
    if (!mxGetField(prhs[0],0,"data")) { mexErrMsgTxt("data missing 'data' field\n"); }
    int8_t *alldata = (int8_t *) mxGetData(mxGetField(prhs[0],0,"data"));
    int bigT = mxGetN(mxGetField(prhs[0],0,"data")); // total length of the data
    int num_subparts = mxGetM(mxGetField(prhs[0],0,"data")); // number of sub-parts

    if (!mxGetField(prhs[0],0,"resources")) { mexErrMsgTxt("data missing 'resources' field\n"); }
    int16_t *allresources = (int16_t *) mxGetData(mxGetField(prhs[0],0,"resources"));

    if (!mxGetField(prhs[0],0,"starts")) { mexErrMsgTxt("data missing 'starts' field\n"); }
    int32_t *starts = (int32_t *) mxGetData(mxGetField(prhs[0],0,"starts"));
    int num_sequences = max(mxGetN(mxGetField(prhs[0],0,"starts")),
            mxGetM(mxGetField(prhs[0],0,"starts")));

    if (!mxGetField(prhs[0],0,"lengths")) { mexErrMsgTxt("data missing 'lengths' field\n"); }
    int32_t *lengths = (int32_t *) mxGetData(mxGetField(prhs[0],0,"lengths"));

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

    // forward messages
    double *all_forward_messages = mxGetPr(prhs[2]);

    //// outputs

    // lhs outputs
    if (nlhs != 1) { mexErrMsgTxt("must have one output\n"); }
    plhs[0] = mxCreateDoubleMatrix(2,bigT,mxREAL);
    double *all_predictions = mxGetPr(plhs[0]);
    Map<Array2Xd,Aligned> predictions(mxGetPr(plhs[0]),2,bigT);

    /* COMPUTATION */

    for (int sequence_index=0; sequence_index < num_sequences; sequence_index++) {
        // NOTE: -1 because Matlab indexing starts at 1
        int sequence_start = ((int) starts[sequence_index]) - 1;
        int T = (int) lengths[sequence_index];

        int16_t *resources = allresources + sequence_start;
        Map<MatrixXd> forward_messages(all_forward_messages + 2*sequence_start,2,T);
        Map<MatrixXd> predictions(all_predictions + 2*sequence_start,2,T);

        predictions.col(0) = initial_distn;
        for (int t=0; t<T-1; t++) {
            predictions.col(t+1) = As.block(0,2*(resources[t]-1),2,2) * forward_messages.col(t);
        }
    }
}

