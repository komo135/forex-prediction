from setuptools import setup

setuo(
  name = "prediction",
  version = 1.0,
  description = 'prediction of stock price or forex price',
  author = 'Nagi',
  author_email = 'komoootv@gmail.com',
  packages = ['prediction'],
  install_requires = ['tensorflow', 'einops', 'yfinance', 'MetaTrader5', 'numpy', 'pandas', 'sklearn']
)
