class Result:
    """Result is an object used to store the LSTM model variation created alongwith the
       datasets created according to time basis like weekly , monthly and the history
    """
    def __init__(self, model, dataframe_x, df_y, df_xtest, df_ytest, history):
        self.model = model
        self.dataframe_x = dataframe_x
        self.df_y = df_y
        self.df_xtest = df_xtest
        self.df_ytest = df_ytest
        self.history = history

        # Result(model, X, y, X_test, y_test, history_lstm)