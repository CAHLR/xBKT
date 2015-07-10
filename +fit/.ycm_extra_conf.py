flags = [
    '-I','/usr/local/include',
    '-I','/Applications/MATLAB_R2013a.app/extern/include/',
    '-I','/opt/local/lib/gcc48/gcc/x86_64-apple-darwin12/4.8.1/include/omp.h',
    '-D','MATLAB_MEX_FILE'
    '-D','NDEBUG',
    '-fno-common',
    '-fopenmp',
    ]

def FlagsForFile(filename):
    return {
            'flags': flags,
            'do_cache': True
            }

