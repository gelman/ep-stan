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
	@echo '--------------------------------------------------------------------------------'


stan-update:
	@echo 'updating Stan submodule'
	git submodule init
	git submodule update --recursive

stan-update/%: stan-update
	@echo 'updating Stan submodule to $*'
	cd stan && git fetch --all && git checkout $* && git pull
