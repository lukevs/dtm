# Dynamic Topic Modeling
### An implementation of Dynamic Topic Modeling using Non-negative Matrix Factorization

Implements the method used [here](http://arxiv.org/abs/1607.03055):
```
Exploring the Political Agenda of the European Parliament Using a
Dynamic Topic Modeling Approach Derek Greene, James P. Cross.
Political Analysis, 2016.
```

### Installation

Install pip requirements with:
```
pip install -r requirements.txt
```

spacy requests downloading the English model with:
```
python3 -m spacy download en
```

Run with:
```
pytnon3 dtm.py <path to JSON file with object specified below>
```

This repo contains an example ipython notebook. To view, start a jupyter server
in this directory using:
```
jupyter notebook
```


### Data

To make it easy to run the model on multiple datasets, input data is defined by a JSON object:
```
{
  "documents": list of Document,
  "windows": list of Window
}
```

where Document is:
```
{
  "text": the text of the document (str),
  "timestamp": the timstamp of the document (number)
}
```

and Window is:
```
{
  "label": label for the window for graphing (str),
  "start": starting timestamp (number, inclusive),
  "end": ending timestamp (number, exclusive)
}
```
