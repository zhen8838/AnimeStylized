CFG = config/default.yml
CODE = none
CKPT = ./logs

all:
	@echo please use \"make train\" or other ...

train:
	python ${CODE} ${CFG}

tensorboard:
	tensorboard --logdir ${CKPT} --samples_per_plugin images=0