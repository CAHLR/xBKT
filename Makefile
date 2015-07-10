MEXFUNCS=+fit/E_step +fit/sample_states +fit/predict_onestep_states +generate/synthetic_data_helper
CXX=g++-4.9
UNAME=$(shell uname)

# these defaults are set for the machines we've been working on
# they may need to change

ifndef MATLABPATH
	ifeq ($(UNAME),Darwin)
		MATLABPATH=/Applications/MATLAB_R2013a.app
	endif
	ifeq ($(UNAME),Linux)
		MATLABPATH=/usr/local/MATLAB/R2014a
	endif
endif

ifndef EIGENPATH
	ifeq ($(UNAME),Darwin)
		EIGENPATH=/usr/local/include
	endif
	ifeq ($(UNAME),Linux)
		EIGENPATH=/usr/include/eigen3
	endif
endif

# nothing should need to change below here (assuming x86_64)
# NOTE: find the flags mex wants by running mex -v somefile.cpp within Matlab

ifeq ($(UNAME),Darwin)
	MEXEXT=mexmaci64
	MEXARCH=maci64
	LDFLAGS=-bundle -Wl,-exported_symbols_list,$(MATLABPATH)/extern/lib/$(MEXARCH)/mexFunction.map
endif
ifeq ($(UNAME),Linux)
	MEXEXT=mexa64
	MEXARCH=glnxa64
	CXXFLAGS=-fPIC -fno-omit-frame-pointer -DMX_COMPAT_32 -D_GNU_SOURCE
	LDFLAGS=-shared -Wl,--version-script,$(MATLABPATH)/extern/lib/$(MEXARCH)/mexFunction.map
endif

ALL=$(addsuffix .$(MEXEXT),$(MEXFUNCS))
OBJS=$(addsuffix .o,$(MEXFUNCS))

MACROS=NDEBUG EIGEN_DONT_PARALLELIZE MATLAB_MEX_FILE
INCLUDES=$(EIGENPATH) $(MATLABPATH)/extern/include
LINKDIRS=$(MATLABPATH)/bin/$(MEXARCH)
LINKLIBS=mex mx mat stdc++

CXXFLAGS+=$(addprefix -I,$(INCLUDES)) $(addprefix -D,$(MACROS)) -O3 -fno-common -fopenmp -march=native
LDFLAGS+=$(addprefix -L,$(LINKDIRS)) $(addprefix -l,$(LINKLIBS)) -fopenmp


all: $(ALL)

%.$(MEXEXT): %.o
	$(CXX) $(LDFLAGS) -o $@ $<

debug: CXXFLAGS+=-DDEBUG -g -fno-openmp
debug: all

clean:
	-rm -f $(ALL) $(OBJS)

