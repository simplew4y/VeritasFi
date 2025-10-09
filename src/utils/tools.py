# tools.py
import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = ""
BASE_URL = 'https://finnhub.io/api/v1'

def get_ipo_info(ticker: str = "LOT") -> dict:
    """Get IPO information for a given ticker"""
    if not ticker:
        return {"error": "Ticker symbol cannot be empty."}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    try:
        url = f"{BASE_URL}/calendar/ipo"
        params = {
            'from': start_date_str,
            'to': end_date_str,
            'token': FINNHUB_API_KEY
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        ipo_calendar = response.json().get('ipoCalendar', [])
        if not ipo_calendar:
            return {"error": f"No IPO data found in the last two years."}
        
        for ipo in ipo_calendar:
            if ipo.get('symbol') == ticker.upper():
                ipo_info = {
                    "companyName": ipo.get('name'),
                    "ticker": ipo.get('symbol'),
                    "ipoDate": ipo.get('date'),
                    "ipoPrice": ipo.get('price'),
                    "shares": ipo.get('numberOfShares'),
                    "exchange": ipo.get('exchange')
                }
                print(f"[IPO INFO TOOL] {ipo_info}")
                return ipo_info
        
        return {"error": f"IPO information for {ticker} not found."}
    
    except Exception as e:
        logger.error(f"Error fetching IPO info: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

def get_stock_price(ticker: str = "ZK") -> dict:
    """Get current stock price for a given ticker"""
    if not ticker:
        return {"error": "Ticker symbol cannot be empty."}
    
    try:
        url = f"{BASE_URL}/quote"
        params = {'symbol': ticker.upper(), 'token': FINNHUB_API_KEY}
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('c') == 0 and data.get('pc') == 0:
            return {"error": f"No price data available for {ticker}."}
        
        stock_price = {
            "ticker": ticker.upper(),
            "currentPrice": data.get('c'),
            "previousClose": data.get('pc'),
            "change": data.get('d'),
            "percentChange": data.get('dp')
        }
        print(f"[STOCK PRICE TOOL] {stock_price}")
        return stock_price
    
    except Exception as e:
        logger.error(f"Error fetching stock price: {e}")
        return {"error": f"An unexpected error occurred: {e}"}