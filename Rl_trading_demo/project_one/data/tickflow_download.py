
import os
from dotenv import load_dotenv
from tickflow import TickFlow
import datetime


load_dotenv()
tickflow_apikey = os.getenv("tickflow_apikey")


tf = TickFlow(api_key=tickflow_apikey)

# 将日期转换为毫秒时间戳
start = int(datetime.datetime(2024, 1, 1).timestamp() * 1000)
end = int(datetime.datetime(2024, 12, 31).timestamp() * 1000)

doupo_etf = "159985.SZ"



