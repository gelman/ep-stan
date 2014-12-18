help:


help:
	@echo '--------------------------------------------------------------------------------'
	@echo 'ep-stan makefile'
	@echo '   Given: *.stan'
	@echo '   Type:  make *'
	@echo ''
	@echo '   Hopefully this will build an EP executable'
	@echo ''
	@echo '   Update Stan submodule targets:'
	@echo '      stan-update:   updates Stan'
	@echo '      stan-update/*: updates Stan to branch *'
	@echo ''
	@echo '   Build the compiler'
	@echo '      bin/stanc'
	@echo '--------------------------------------------------------------------------------'


stan-update:
	@echo 'updating Stan submodule'
	git submodule init
	git submodule update --recursive

stan-update/%: stan-update
	@echo 'updating Stan submodule to $*'
	cd stan && git fetch --all && git checkout $* && git pull

STANAPI_HOME ?= stan/
EIGEN ?= $(STANAPI_HOME)lib/eigen_3.2.2
BOOST ?= $(STANAPI_HOME)lib/boost_1.54.0
GTEST ?= $(STANAPI_HOME)lib/gtest_1.7.0

EXE :=
EP_MAIN := src/main.cpp
CXX := clang++
CXXFLAGS := -DBOOST_RESULT_OF_USE_TR1 -DBOOST_NO_DECLTYPE -DBOOST_DISABLE_ASSERTS -I src -I $(STANAPI_HOME)src -isystem $(EIGEN) -isystem $(BOOST) -Wall -pipe -DEIGEN_NO_DEBUG
O_STANC := 0
LDLIBS_STANC := -Lbin -lstanc

TEMPLATE_INSTANTIATION := $(shell find stan/src/stan/gm -type f -name '*_inst.cpp')
TEMPLATE_INSTANTIATION += $(shell find stan/src/stan/gm -type f -name '*_def.cpp')
TEMPLATE_INSTANTIATION := $(TEMPLATE_INSTANTIATION:stan/src/%.cpp=bin/%.o)
bin/libstanc.a : $(TEMPLATE_INSTANTIATION)
	@mkdir -p $(dir $@)
	$(AR) -rs bin/libstanc.a $(TEMPLATE_INSTANTIATION)

$(TEMPLATE_INSTANTIATION) : bin/stan/%.o : stan/src/stan/%.cpp
	@mkdir -p $(dir $@)
	$(COMPILE.cc) -O$(O_STANC) $(OUTPUT_OPTION) $<


.PRECIOUS: bin/stanc$(EXE)
bin/stanc$(EXE) : bin/stanc.o bin/libstanc.a
	@mkdir -p $(dir $@)
	$(LINK.cc) -O$(O_STANC) $(OUTPUT_OPTION) $< $(LDLIBS_STANC)


build: hierarchical_logistic


bin/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	$(COMPILE.cc) -O$O $(OUTPUT_OPTION) $<

%.hpp: %.stan bin/stanc$(EXE)
	@echo ''
	@echo '--- Translating Stan model to C++ code ---'
	bin/stanc$(EXE) $< --o=$@

$(patsubst %.stan,%,$(wildcard $(addsuffix .stan,$(MAKECMDGOALS)))) : %$(EXE) : %.hpp
	@echo ''
	@echo '--- Linking C++ model ---'
	$(LINK.cc) -O$O $(OUTPUT_OPTION) $(EP_MAIN) -include $<

.PHONY: clean
clean:

.PHONY: clean-all
clean-all: clean
	$(RM) -r bin

-include make/local
