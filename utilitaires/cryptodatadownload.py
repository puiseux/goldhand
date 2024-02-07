import pandas as pd
def download(url="https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"):
    # df = pd.read_csv(filepath, skiprows=1)  # we use skiprows parameter because first row contains our web address
    df = pd.read_csv(filepath)  # we use skiprows parameter because first row contains our web address

    return df
if __name__ == '__main__':
    filepath = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv"
    filepath = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_minute.csv"
    filepath = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
    df = download(filepath)
    print(df.head(15))
