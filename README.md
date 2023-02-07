# chat-gpt-from-scratch

This is an implemntation of Transformer model from [Attention is all you need](https://arxiv.org/abs/1706.03762) paper. I use Jupyter Notebook to learn its architecture and training in detail. Then, modified the codes from [nanoGPT repo](https://github.com/deep-deep-learning/nanoGPT) to experiment it on Shakespeare data. With the hyperparameter setting in `train.py`, I was able to generate following text:

```
LADY GREY ANNE:
Good lords, you, that he were you must well.
What is your house of this day?

Provost:
You shall, I'll hear me here!

LADY ANNE:
My lord, I die, if the Duke of your grace.

LADY ANNE:
I'll here, I'll tell thee to the shame at the Tower!
```
 
## dependencies
- PyTorch
- NumPy
- requests: for dataset download
- ticktoken: for tokenizer

    ```$ pip install -r requirements.txt```

## Shakespeare data

Data sample:
```
VIRGILIA:
No, at a word, madam; indeed, I must not. I wish
you much mirth.

VALERIA:
Well, then, farewell.

MARCIUS:
Yonder comes news. A wager they have met.

LARTIUS:
My horse to yours, no.

MARCIUS:
'Tis done.

LARTIUS:
Agreed.
```

The data can be prepared by:

    $ python data/data.py

The original data is in `notebooks/input.txt`

## train

Train the model

    $ python train.py

## sample

Sample from the best trained model

    $ python sample.py
