CFG = config/default.yml
CODE = none

all:
	@echo please use \"make train\" or other ...

train:
	python ${CODE} ${CFG}