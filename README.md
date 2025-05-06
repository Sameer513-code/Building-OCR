# Building-OCR


### Classification Report
You'll see something like this:
| Term          | Simple Meaning                                         |
| ------------- | ------------------------------------------------------ |
| **Precision** | Of all predicted as "this label", how many were right? |
| **Recall**    | Of all actual "this label", how many did we catch?     |
| **F1-score**  | A balance of precision & recall (ideal = 1.0)          |
| **Support**   | Number of test images for that label                   |


Example:<br>
If Label: 3 has low *precision*, it means the model often thinks something is a 3 when it’s not.<br>
If Label: 8 has low *recall*, the model misses a lot of actual 8s (e.g., it confuses them with 3s).<br>

### Confusion Matrix
It’s a big table like this:
| True\Pred | +   | -  | 3  | 8  |
| --------- | --- | -- | -- | -- |
| **+**     | 110 | 5  | 2  | 3  |
| **-**     | 3   | 88 | 6  | 3  |
| **3**     | 1   | 4  | 54 | 31 |
| **8**     | 2   | 1  | 22 | 70 |

🎯 What to Look For:
 - Diagonal values (↘) are correct predictions. Higher = better.
 - Off-diagonal values are misclassifications.
<br>
Example Analysis: <br>
If 31 “3”s were predicted as “8” → that’s a big confusion. <br>
