from src.figures import *
import config

# config.py:
# plotly_username = "PLOTLY_USER_NAME"
# plotly_api_key = "PLOTLY_API_KEY"


create_plotly_figure(
    username = config.plotly_username, 
    api_key  = config.plotly_api_key, 
    input    = "./figures/pci.csv",
    output   = "pci_v0.2.0"
)
