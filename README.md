# Chetverikov method for time-series analysis

Provided code implements Chetverikov algorithm to calculate:

1. Preliminary, first and second trend assesments
2. First and second assesments of seasonality
3. Residual component
4. The intensity coefficient of the seasonal wave

## Installation

### Manual

Install package through git:

    pip install git+https://github.com/Dragon066/chetverikov.git

**OR**

Clone this repository:

    git clone https://github.com/Dragon066/chetverikov.git

Install requirements:

    pip install -r requirements.txt

## Usage

This package provides class `Chetverikov`, that includes several methods for time-series analysis.

To fit model, use `.fit(y, L, rolling)` method:
- `y` - initial series;
- `L` - number of periods;
- `rolling` - size of rolling window for computing second trend assesment;

To get results, you can use:
- `.summary()` method to create summary table;
- `.plot(chart)` method to plot some data. Available values for `chart` are `trend_all`, `trend`, `season`, `resid`, `k`.

### Example

```python
import numpy as np
from chetverikov import Chetverikov

np.random.seed(0)

trend = np.arange(1, 49) + np.random.rand(48) * 10

season = np.broadcast_to(np.random.rand(1, 12), (4, 12)).reshape(-1) * 10

data = trend + season

model = Chetverikov().fit(data)

print(model.summary())

# For plotting
# model.plot("season")
```
