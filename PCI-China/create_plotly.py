from src.figures import *
import config
import plotly

# config.py:
# plotly_username = "PLOTLY_USER_NAME"
# plotly_api_key = "PLOTLY_API_KEY"

plotly.tools.set_credentials_file(
    username=config.plotly_username, 
    api_key=config.plotly_api_key
)

fig = create_plotly_figure(input = "./figures/pci.csv")

plotly.plotly.iplot(fig, filename = "pci_v0.7.0", auto_open=True,show_link=True)

