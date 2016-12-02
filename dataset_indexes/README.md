# Dataset Saturation Indexes

These files hold compressed json files of the following format.
Please look at `dataset_utilities/create_dataset_saturation_index.py` for information on how to read and write from the dataset.

```python
import ujson, gzip
f = gzip.open('saturation_values.json.gz', 'rt')
saturation_index = ujson.load(f)
saturation_index['n03982430/n03982430_22677.JPEG']
# 0.5078
```
