# Neural-EDU-Segmentation
A toolkit for segmenting Elementary Discourse Units (clauses).
We implement it as is described in our EMNLP paper: [Toward Fast and Accurate Neural Discourse Segmentation]()


### Requirements
- Python 3.5
- Tensorflow>=1.5.0
- allennlp>=0.4.2
- See `requirements.txt` for the full list of packages

### Data

We cannot provide the complete [RST-DT corpus](https://catalog.ldc.upenn.edu/products/LDC2002T07) due to the LDC copyright.
So we only put several samples in `./data/rst/` to test the our code and show the data structure.

If you want to train or evaluate our model on RST-DT, you need to download the data manually and put it in the same folder. Then run the following command to preprocess the data and create the vocabulary:

```
python run.py --prepare
```


### Evaluate the model on RST-DT:

We provide the vocabulary and a well-trained model in the `./data/` folder. You can evaluate the performance of this model after preparing the RST-DT data as mentioned above:

```
python run.py --evaluate --test_files ../data/rst/preprocessed/test/*.preprocessed
```

The performance of current model should be as follows:
```
{'precision': 0.9176470588235294, 'recall': 0.975, 'f1': 0.9454545454545454}
```

Note that this is slightly better than the results we reported in the paper, since we re-trained the model and there is some randomness here.

### Train a new model

You can use the following command to train the model from scratch:

```
python run.py --train
```

Hyper-parameters and other training settings can be modified in `config.py`.

### Segmenting raw text into EDUs

[Coming soon]


### Citation

Please cite the following paper if you use this toolkit in your work:

```
@inproceedings{wang2018edu,
  title={Toward Fast and Accurate Neural Discourse Segmentation},
  author={Yizhong Wang, Sujian Li and Jingfeng Yang},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)},
  year={2018}
}
```
