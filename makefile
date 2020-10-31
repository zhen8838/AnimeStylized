CFG = config/default.yml
CODE = None
LOGDIR = ./logs
CKPT = None
EXTRA = None

all:
	@echo please use \"make train\" or other ...

train:
	python ${CODE} --config ${CFG} --stage fit --ckpt ${CKPT}

test:
	python ${CODE} --config ${CFG} --stage test --ckpt ${CKPT}

infer:
	python ${CODE} --config None --stage infer --ckpt ${CKPT} --extra ${EXTRA}

tensorboard:
	tensorboard --logdir ${LOGDIR} --samples_per_plugin images=0