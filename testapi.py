from okx.MarketData import MarketAPI

market = MarketAPI()
result = market.get_tickers('SPOT')
print(result)
