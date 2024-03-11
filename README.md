
# Value Investor

## Problem statement:

We are a portfolio investment company and we make investments in the emerging markets around the world. Our company profits by investing in profitable companies, buying, holding and selling company stocks based on value investing principles.


Our goal is to establish a robust intelligent system to aid our value investing efforts using stock market data. We make investment decisions and based on intrinsic value of companies and do not trade on the basis of daily market volatility. Our profit realization strategy typically involves weekly, monthly and quarterly performance of stocks we buy or hold.


## Data Description:


Here the data taken is for MICROSOFT stock from yahoo finance using python API.


## Goal(s):


Predict stock price valuations on a daily, weekly and monthly basis. Recommend BUY, HOLD, SELL decisions. Maximize capital returns, minimize losses. Ideally a loss should never happen. Minimize HOLD period.


## Success Metrics:


Evaluate on the basis of capital returns. Use Bollinger Bands to measure your systems effectiveness.


# Method and Results
## Models
I built several models for this Time Series Analysis: (MA)moving average, ARIMA, SARIMAX, exponential smoothing all types: SES, DES, TES, FB Prophet and finally neural network based LSTM sequential. I trained each model for daily, weekly and monthly, and with different lookback windows. I then gathered test scores for each (root mean squared error) in order to compare models. I also have created an /evaluate API in Python Flask for the same wherein based choosing the stock and the lookback window you can evaluate and look at the rmse score for each which is returned by the API.

## Results
Amongst all the models give above, the best performance could be noted for LSTM followed by Prophet. So, for LSTMs I also computed for different lookback periods like daily, weekly, monthly to check which works best and gives us the best selling recommendataion.

## Investments
Based on these trained models, I then built an investment function wherein I first made the Bollinger bands for the predictions, and then defined buy and sell strategies based whether the actual data was above (sell), within (hold) or below (buy) the bollinger bands.
This is depicted in the Flask webpage backed by the /predict API wherein you can choose the Ticker symbol for Microsoft and select the duration and it will return a detailed Bollinger band based signalling graph with upticks for selling the stock and downticks for buying the stock at that point in time returning the maximum earnable profit for that period.

## Project Organization
<p><small>The project is a <i>Flask App</i> based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## Run
Run project by installing the <b>requirements.txt</b> dependencies and then executing the command `python app.py`