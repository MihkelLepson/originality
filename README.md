# originality
The python package for assessing texts originality. The texts orginality is assessed by comparing the target text with reference texts. To compare the target with references the longest common subsequence is calculated. To speed up the calculation the CUDA capable GPU is required.

## Example
To calculate the originality, the texts must be first tokenized. Currently empty strings are not allowed.

```
from originality import lcs

targets = [[1,2,3],[20,30,40,50]]
references = [[1, 20, 30],[40,50,4,6,1],[23,5,1,2,5,2,5,1,3]]

print(lcs.check_originality(targets, references))

[[1 1 3]
 [2 2 0]]
```
