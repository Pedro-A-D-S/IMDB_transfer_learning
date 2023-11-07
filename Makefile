init:
	poetry env use python3.8
	poetry shell
	poetry install

create: utils/create_folders.sh
	sh utils/create_folders.sh

glove: utils/get_glove_embeddings.sh
	sh utils/get_glove_embeddings.sh
