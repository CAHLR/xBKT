#include <stdint.h>
#include <cstring>
#include <Eigen/Core>
#include <omp.h>
#include "mex.h"

using namespace Eigen;
using namespace std;

// TODO if we aren't outputting stateseqs, don't write it to memory (just need
// t and t+1), so we can save a stack array for each HMM at the cost of a
// branch

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{

    /* SETUP */
    if (nrhs != 6) { mexErrMsgTxt("wrong number of arguments\n"); }

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
    int num_sequences = max(mxGetM(mxGetField(prhs[0],0,"starts")),
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
    MatrixXd ATs(2,2*num_resources);
    for (int n=0; n<num_resources; n++) {
        ATs.block(0,2*n,2,2) = As.block(0,2*n,2,2).transpose();
    }

    Array2Xd Bn(2,2*num_subparts);
    for (int n=0; n<num_subparts; n++) {
        Bn.col(2*n) << 1-guess[n], slip[n]; // incorrect
        Bn.col(2*n+1) << guess[n], 1-slip[n]; // correct
    }

    // temp
    double temp = mxGetScalar(prhs[5]);
    Bn = (Bn.log()/temp).exp();

    //// outputs

    // rhs outputs
    Map<Array2Xd,Aligned> all_trans_counts((double *) mxGetData(prhs[2]),2,2*num_resources);
    all_trans_counts.setZero();
    Map<Array2Xd,Aligned> all_emission_counts((double *) mxGetData(prhs[3]),2,2*num_subparts);
    all_emission_counts.setZero();
    Map<Array2d,Aligned> all_initial_counts((double *) mxGetData(prhs[4]));
    all_initial_counts.setZero();

    // lhs outputs
    Map<Array2Xd,Aligned> beta_out(NULL,2,bigT);
    int8_t *all_stateseqs = NULL;
    double s_total_loglike = 0;
    double *total_loglike = &s_total_loglike;
    switch (nlhs)
    {
        case 3:
            plhs[2] = mxCreateDoubleMatrix(2,bigT,mxREAL);
            new (&beta_out) Map<Array2Xd,Aligned>(mxGetPr(plhs[2]),2,bigT);
        case 2:
            plhs[1] = mxCreateNumericMatrix(1,bigT,mxINT8_CLASS,mxREAL);
            all_stateseqs = (int8_t *) mxGetData(plhs[1]);
        case 1:
            plhs[0] = mxCreateDoubleScalar(0.);
            total_loglike = mxGetPr(plhs[0]);
    }

    /* COMPUTATION */
    Eigen::initParallel();
    /* omp_set_dynamic(0); */
    /* omp_set_num_threads(6); */
    #pragma omp parallel
    {
        double s_trans_counts[2*2*num_resources], s_emission_counts[2*2*num_subparts],
               s_init_counts[2], loglike;
        memset(s_trans_counts,0,sizeof(s_trans_counts));
        memset(s_emission_counts,0,sizeof(s_emission_counts));
        memset(s_init_counts,0,sizeof(s_init_counts));

        Map<Array2Xd,Aligned> trans_counts(s_trans_counts,2,2*num_resources);
        Map<Array2Xd,Aligned> emission_counts(s_emission_counts,2,2*num_subparts);
        Map<Array2d,Aligned> init_counts(s_init_counts,2,1);

        loglike = 0;

        int num_threads = omp_get_num_threads();
        int blocklen = 1 + ((num_sequences - 1) / num_threads);
        int sequence_idx_start = blocklen * omp_get_thread_num();
        int sequence_idx_end = min(sequence_idx_start+blocklen,num_sequences);

        for (int sequence_index=sequence_idx_start; sequence_index < sequence_idx_end; sequence_index++) {
            // NOTE: -1 because Matlab indexing starts at 1
            int32_t sequence_start = starts[sequence_index] - 1;
            int32_t T = lengths[sequence_index];

            int8_t *data = alldata + num_subparts*sequence_start;
            int16_t *resources = allresources + sequence_start;

            //// likelihoods
            double s_likelihoods[2*T];
            Map<Array2Xd,Aligned> likelihoods(s_likelihoods,2,T);
            likelihoods.setOnes();
            for (int t=0; t<T; t++) {
                for (int n=0; n<num_subparts; n++) {
                    if (data[n+num_subparts*t] != 0) {
                        likelihoods.col(t) *= Bn.col(2*n + (data[n+num_subparts*t] == 2));
                    }
                }
            }

            //// backward messages
            // NOTE: these aren't the standard betas; I'm including the
            // evidence potential at time t in beta.col(t) as well so we don't
            // have to touch the likelihoods on the second pass
            double norm;
            double s_beta[2*T];
            Map<ArrayXXd,Aligned> beta(s_beta,2,T);
            beta.col(T-1) = likelihoods.col(T-1);
            norm = beta.col(T-1).sum();
            beta.col(T-1) /= norm;
            loglike += log(norm);
            for (int t=T-2; t>=0; t--) {
                beta.col(t) = (ATs.block(0,2*(resources[t]-1),2,2) *
                    beta.col(t+1).matrix()).array() * likelihoods.col(t);
                norm = beta.col(t).sum();
                beta.col(t) /= norm;
                loglike += log(norm);
            }
            loglike += log((beta.col(0) * initial_distn).sum());

            //// sample forwards and statistic counting
            int8_t stateseq[T];
            int t=0;
            Array2d nextstate_unsmoothed = initial_distn;
            Array2d nextstate_distr = nextstate_unsmoothed * beta.col(t);
            stateseq[t] = nextstate_distr(0) < nextstate_distr.sum()
                * (((double) random() )/((double) RAND_MAX ));
            nextstate_unsmoothed = As.col(2*(resources[t]-1)+stateseq[t]);
            // initial stats
            s_init_counts[stateseq[t]] += 1.0;
            // singleton stats
            for (int n=0; n<num_subparts; n++) {
                if (data[n+num_subparts*t] != 0) {
                    s_emission_counts[2*(2*n+(data[n+num_subparts*t]==2))+stateseq[t]] += 1.0;
                }
            }
            for (t=1; t<T; t++) {
                Array2d nextstate_distr = nextstate_unsmoothed * beta.col(t);
                stateseq[t] = nextstate_distr(0) < nextstate_distr.sum()
                    * (((double) random() )/((double) RAND_MAX ));
                nextstate_unsmoothed = As.col(2*(resources[t]-1)+stateseq[t]);

                // singleton stats
                for (int n=0; n<num_subparts; n++) {
                    if (data[n+num_subparts*t] != 0) {
                        s_emission_counts[4*n+2*(data[n+num_subparts*t]==2)+stateseq[t]] += 1.0;
                    }
                }

                // pairwise stats
                s_trans_counts[4*(resources[t-1]-1)+2*stateseq[t-1]+stateseq[t]] += 1.0;
            }

            switch (nlhs)
            {
                case 3:
                    beta_out.block(0,sequence_start,2,T) = beta;
                case 2:
                    memcpy(all_stateseqs+sequence_start,stateseq,sizeof(stateseq));
            }
        }

        #pragma omp critical
        {
            all_trans_counts += trans_counts;
            all_emission_counts += emission_counts;
            all_initial_counts += init_counts;
            *total_loglike += loglike;
        }
    }
}

