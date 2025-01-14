# GOTLoc: General Outdoor Text-based Localization Using Scene Graph Retrieval with OpenStreetMap
[Video](), [Paper]()

## Description
This is an implentation of *"GOTLoc: General Outdoor Text-based Localization Using Scene Graph Retrieval with OpenStreetMap"* which indicates a general outdoor text-based localization using scene graph retrieval with OpenStreetMap.

## Prerequisite
### Install the vectorDB (Milvus)
<!-- For the scene graph candidates extraction, the vectorDB is necessary.
To install the vectorDB. Follow the instruction [link](https://github.com/milvus-io/milvus).
If you can't install the vectorDB for any reasons. Skip the process and proceed the codes without candidates extraction step.
However, skipping the extraction step makes the speed of the process much slower. -->
The vectorDB is required for extracting scene graph candidates. To install it, follow the instructions at this [link](https://github.com/milvus-io/milvus). If you're unable to install the vectorDB for any reason, you can skip this step and continue with the code without the candidate extraction. However, skipping this step will significantly slow down the process.

## Installation
### Setting an environments
```bash
conda env create -f environment.yml
conda activate GOTLoc
```
### Install the NLTK
Install the trained pipeline for *word2vec* embeddings.
```bash
python -m spacy download en_core_web_lg
```
### Setup directories
```bash
bash setup_directories.sh
```

## Dataset
### Download
- [Download](https://drive.google.com/drive/folders/1oLksAHJl-AUjUM-LIVP5e3i9wMGqhxyl?usp=sharing) Download scene graphs and model checkpoints.
- Move the downloaded data to the `data` directory.
- Unzip the downloaded data.

## Train
You can modify the training arguments, which are specified in the `config.py` file.
```bash
python train.py
```

## Evaluation
To modify the evaluation arguments, please consult the `config.py`. file. Additionally, if you have not installed vectorDB (Milvus) as outlined in [Install the vectorDB (Milvus)](#Install-the-vectorDB-(Milvus)), set this value to *False*.
```bash
python eval.py
```

## Visualization
To modify the target scene graph for visualization, please check the *visualization_graphs_file_name* and *visualization_graph_index* in the `config.py` file.
```bash
python visualize_graph.py
```

## Citation
```
```

## Acknowledgements
The codes and datasets in this repository are based on [Where am I?](https://github.com/jiaqchen/whereami-text2sgm), [Milvus](https://github.com/milvus-io/milvus), and [Text2Pos](https://github.com/mako443/Text2Pos-CVPR2022). Thanks to the authors of these codes and datasets.

## License

Copyright 2025, Donghwi Jung, Keonwoo Kim, Seong-Woo Kim, Autonomous Robot Intelligence Lab, Seoul National University.

This project is free software made available under the MIT License. For details see the LICENSE file.