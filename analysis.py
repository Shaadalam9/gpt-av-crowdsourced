import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import common
from custom_logger import CustomLogger
from logmod import logs
from ollama import OllamaClient


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# set template for plotly output
template = common.get_configs('plotly_template')

client = OllamaClient()

ans = client.generate_simple_text(sentence="Hi, I am 45 years old", model="deepseek-r1:1.5b")
