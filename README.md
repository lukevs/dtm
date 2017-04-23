# Dynamic Topic Modeling
### An implementation of Dynamic Topic Modeling using Non-negative Matrix Factorization

Implements the method used [here](http://arxiv.org/abs/1607.03055):
```
Exploring the Political Agenda of the European Parliament Using a 
Dynamic Topic Modeling Approach Derek Greene, James P. Cross. 
Political Analysis, 2016.
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
