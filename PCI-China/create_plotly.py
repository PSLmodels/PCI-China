from src.figures import *
import config
import chart_studio 

# config.py:
# plotly_username = "PLOTLY_USER_NAME"
# plotly_api_key = "PLOTLY_API_KEY"

chart_studio.tools.set_credentials_file(
    username=config.plotly_username, 
    api_key=config.plotly_api_key
)

fig = create_plotly_figure(input = "./figures/pci.csv")

chart_studio.plotly.plot(fig, filename = "pci_v_2022_02_24", auto_open=True)

