### Original Dataset Format ###

Sample Entry:
~~~
{
  "24198545_gab": {
    "post_id": "24198545_gab",
    "annotators": [
      {
        "label": "hatespeech",
        "annotator_id": 4,
        "target": ["African"]
      },
      {
        "label": "hatespeech",
        "annotator_id": 3,
        "target": ["African"]
      },
      {
        "label": "offensive",
        "annotator_id": 5,
        "target": ["African"]
      }
    ],
    "rationales":[
    [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    "post_tokens": ["and","this","is","why","i","end","up","with","nigger","trainee","doctors","who","can","not","speak","properly","lack","basic","knowledge","of","biology","it","truly","scary","if","the","public","only","knew"]
  }
}
~~~

:small_blue_diamond:post_id : Unique id for each post<br/>
:small_blue_diamond:annotators : The list of annotations from each annotator<br/>
:small_blue_diamond:annotators[label] : The label assigned by the annotator to this post. Possible values: [Hatespeech, Offensive, Normal]<br/>
:small_blue_diamond:annotators[annotator_id] : The unique Id assigned to each annotator<br/>
:small_blue_diamond:annotators[target] : A list of target community present in the post<br/>
:small_blue_diamond:rationales : A list of rationales selected by annotators. Each rationales represents a list with values 0 or 1. A value of 1 means that the token is part of the rationale selected by the annotator. To get the particular token, we can use the same index position in "post_tokens"<br/>
:small_blue_diamond:post_tokens : The list of tokens representing the post which was annotated<br/>

----------------------------

Simplify the above dataset format into three pickle files corresponding to three datasets: train, val and test, where each of them has the following format `(List_of_tokens, Labels)`

How to load the pickle file:
```
import pickle

with open ('trainHateXplain', 'rb') as fp:
    train = pickle.load(fp)
```
