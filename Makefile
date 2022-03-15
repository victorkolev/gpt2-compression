docker-build:
	docker build . -f Dockerfile -t gpt2-compression

run:
	docker run --gpus all -it --net=host -v `pwd`:/gpt2-compression gpt2-compression /bin/bash
