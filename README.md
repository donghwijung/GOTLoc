# GOTLoc: General Outdoor Text-based Localization Using Scene Graph Retrieval with OpenStreetMap
[Video](), [Paper]()

## Description
This is an implentation of *"GOTLoc: General Outdoor Text-based Localization Using Scene Graph Retrieval with OpenStreetMap"* which indicates a general outdoor text-based localization using scene graph retrieval with OpenStreetMap.

## Prerequisite
### Install the vectorDB (Milvus)
For the scene graph candidates extraction, the vectorDB is necessary.
To install the vectorDB. Follow the instruction [link](https://github.com/milvus-io/milvus).
If you can't install the vectorDB for any reasons. Skip the process and proceed the codes without candidates extraction step.
However, skipping the extraction step makes the speed of the process much slower.

## Installation
### Setting an environments
```bash
conda env create -f environment.yml
conda activate GOTLoc
```
### Install the NLTK
Install the trained pipeline for *word2vec* embeddings
```bash
python -m spacy download en_core_web_lg
```
### Setup directories
```bash
bash setup_directories.sh
```

## Dataset
### Download
<!-- - [Download]() KITTI360pose dataset -->
- [Download](https://drive.google.com/drive/folders/1oLksAHJl-AUjUM-LIVP5e3i9wMGqhxyl?usp=sharing) download scene graphs and model checkpoints
- Place the downloaded datasets to the *data* directory
- Unzip the downloaded dataset
<!-- ### Data processing
Generate an OSM scene graph
```bash
python generate_osm_scene_graph.py
# ex) python .py --data_id 3
```
Generate a text scene graph
```bash
python generate_text_scene_graph.py
# ex) python .py --data_id 3 -->
```

## Train
```bash
python train.py
```

## Evaluation
```bash
python eval.py
```

## Citation
```
```

## Acknowledgements
The codes and datasets in this repository are based on [Where am I?](https://github.com/jiaqchen/whereami-text2sgm), [Milvus](https://github.com/milvus-io/milvus), and [Text2Pos](https://github.com/mako443/Text2Pos-CVPR2022). Thanks to the authors of these codes and datasets.

## License

Copyright 2025, Donghwi Jung, Keonwoo Kim, Seong-Woo Kim, Autonomous Robot Intelligence Lab, Seoul National University.

This project is free software made available under the MIT License. For details see the LICENSE file.