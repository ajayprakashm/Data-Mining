# Data-Mining

### Problem statement:
Here, objective is to mine the dataset "groceries.csv" for pairwise association rules. Final results should be a form of dictionary that meet certain conditions and are listed below:
* The keys are pairs (a,b) where a and b are item names (as strings). The values are the corresponding confidence scores.
* Only include rules a=>b where item 'a' occurs at least MIN_COUNT times and conf(a=>b) is at least THRESHOLD. 

THRESHOLD = 0.5 (Confidence threshold)

MIN_COUNT = 10 (Items appearing at least 'MIN_COUNT' times)

### Results:
Given problem is solved using mlxtend library and python code is atached.
