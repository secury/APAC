# Reinforcement Learning for Control with Multiple Frequencies

This repository is the official implementation of [Reinforcement Learning for Control with Multiple Frequencies](http://ailab.kaist.ac.kr/papers/pdfs/LLK2020.pdf). 

![apac_model](https://raw.githubusercontent.com/secury/APAC/main/thumbnail.png)


## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

## Training and Evaluation

To train the model(s) in the paper, run this command:

```
python main.py --train_env_name=apwalker_421421 --eval_env_name=apwalker_421421
```

## Results

![apac_result](https://raw.githubusercontent.com/secury/APAC/main/apac_result.png)
