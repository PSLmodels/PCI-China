import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd 
import datetime
import plotly.io as pio
import plotly

plotly.tools.set_credentials_file(username='ctszkin', api_key='ECoP2DnHVGOPDdYRSZSW')

data = pd.read_csv("./visualization/window_5_years_quarterly/pci.csv")

df = pd.DataFrame({
        "y" : data.pci,
        "x" : pd.to_datetime(data.date),
        "label" : pd.to_datetime(data.date).apply(lambda x: x.date()).apply(lambda x :  x.strftime("%Y-%b")) + data.pci.apply(lambda x: "<br>PCI for China: " +  str(round(x,4)) )
})
df.sort_values(by='x')

y = list(df.y)
x =  list(df.x)

# events_list = [{'date': pd.to_datetime('1953-01-01'), 'text' : "1953 first Five-Year Plan" },
#     {'date': pd.to_datetime('1958-05-16'), 'text' : "1958 Great Leap Forward" },
#     {'date': pd.to_datetime('1966-05-16'), 'text' : "1966 Cultural Revolution" },
#     {'date': pd.to_datetime('1976-09-09'), 'text' : "1976 Hua takes over" },
#     {'date': pd.to_datetime('1978-12-18'), 'text' : "1978 reform program starts" },
#     {'date': pd.to_datetime('1989-06-04'), 'text' : "1989 Tiananmen Sq. protests" },
#     {'date': pd.to_datetime('1993-11-11'), 'text' : "1993 reform speed-up" },
#     {'date': pd.to_datetime('2005-10-08'), 'text' : "2005 reform slow-down" },
#     {'date': pd.to_datetime('2008-11-05'), 'text' : "2008 stimulus package" },
#     {'date': pd.to_datetime('2013-04-22'), 'text' : "2013 revive\nMaoism" },
#     {'date': pd.to_datetime('2013-09-11'), 'text' : "2013 renew\nreform program" },
#     {'date': pd.to_datetime('2015-11-10'), 'text' : "2015\nsupply-\nside\nstructural\nreform" }]

event_d = pd.to_datetime([
    '1953-01-01',
    '1958-05-16',
    '1966-05-16',
    '1976-09-09',
    '1978-12-18',
    '1989-06-04',
    '1993-11-11',
    '2005-10-08',
    '2008-11-05',
    '2013-04-22',
    '2013-09-11',
    '2015-11-10'
])

event_text = [
    "1953-Jan: First Five-Year Plan<br><br>Soviet-style institution put in place;<br>investment in heavy and defense-oriented industries surges.", 
    "1958-May: Great Leap Forward<br><br>Massive transformation of resources from agriculture to industry;<br>rapid expansion of steal production.", 
    "1966-May: Cultural Revolution<br><br>Covert operation to unseat Mao's opponents;<br>investment and industrial construction surge;<br>consumption restrained.",
    "1976-Sep: Hua takes over<br><br>Mao's death, followed by Hua's ascendancy,<br>purge of Gang of Four,<br>and resumption of massive investment push.",
    "1978-Dec: Reform program starts<br><br>Wide-ranging reassessment of command economy;<br>rural reforms; decentralization of powers in industry;<br>markets allowed; trade surges.",
    "1989-Jun: Tiananmen Sq protests<br><br>Students protests,<br>followed by conservative ascendancy,<br>and campaigns against market-oriented reforms.",
    "1993-Nov: Reform speed-up<br><br>Broad reform program kicked off;<br>restrictive macroeconomic policy;<br>price stabilization; fiscal reforms;<br>market unification; economic openness;<br>SOE restructuring.",
    "2005-Oct: Reform slow-down<br><br>Cushions impact of reforms on ''losers:''<br>reducing income inequality;<br>improving access to care and education;<br>extending social security,<br>moderating environment impact of growth.",
    "2008-Nov: stimulus package<br><br>Four trillion yuan economic stimulus package<br>as policy response to global financial crisis.",
    "2013-Apr: Revive Maoism<br><br>Orders officials to fight ''subversive currents,''<br>incl ''ardently market-friendly neo-liberalism.''",
    "2013-Sep: Renew reform program<br><br>Calls for re-commitment to market reforms while<br>retaining emphasis on support for state sector;.",
    "2015-Nov: Supply-side structural reform<br><br>Government-led efforts to reduce excess<br>industrial capacity, esp. steel and coal."
]




label = list(df.label)

trace = go.Scatter(x = x, y = y, text=df.label,hoverinfo ="text" )
data = []
data.append(trace)
data.append(go.Scatter(x=event_d, y = [.4] * 12 , text =event_text  ,hoverinfo ="text", mode='markers', marker=dict(opacity= 0)  ))

shapes = []
for i in event_d:
    shapes.append( dict(
        type = "line",
        line = dict(color = "grey", width = 2.5),
        xref = "x",
        yref = "y",
        x0 = i,
        x1 = i,
        y0 = 0, 
        y1 = 0.4
    ))


layout = dict(
    showlegend = False, 
    hovermode = "x",
    hoverdistance = 16,
    spikedistance = -1,
    yaxis = dict(title = "Quarterly PCI for China"),
    margin=go.layout.Margin(
        l=50,
        r=0,
        b=50,
        t=50,
        pad=4
    ),
    shapes = shapes, 
        hoverlabel = dict(
        font = dict(size = 12, color = "white")
    ),
    xaxis = dict(
        title = "Year", 
        type = "date",
        range = ["1950-01-01", "2022-01-01"],
        # dtick = "M120",
        # tickvals = [1951, 1961, 1971, 1981, 1991, 2001, 2011, 2021], 
        showgrid=False, 
        spikethickness = 2,
        spikecolor ="black",
        showspikes = True,
        spikedash="dot",
        spikemode = "toaxis+across+marker",
        spikesnap = "cursor",
        rangeselector = dict(
            visible = True,
            buttons = list([
                dict(count = 1, 
                    label = "1 yr", 
                    step = "year",
                    stepmode = "backward"),
                dict(count = 5, 
                    label = "5 yr", 
                    step = "year",
                    stepmode = "backward"),
                dict(count = 10, 
                    label = "10 yr", 
                    step = "year",
                    stepmode = "backward"),
                dict(count = 20, 
                    label = "20 yr", 
                    step = "year",
                    stepmode = "backward"),
                dict(step='all')
            ])
        ),
        rangeslider = dict(
            thickness=0.2,
            visible = True,
            yaxis = dict(rangemode = "fixed", range = [0,0.5])
        )
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = 'pci_v0.2.0', auto_open=True,show_link=False)


# fig = go.FigureWidget(data=data, layout=layout)
# fig.layout.on_change(zoom, 'xaxis.range')
# def zoom(layout, xrange):
#     start = datetime.datetime.strptime(fig.layout.xaxis.range[0][0:10],"%Y-%M-%d").date()
#     end = datetime.datetime.strptime(fig.layout.xaxis.range[1][0:10],"%Y-%M-%d").date()
#     in_view = df[ (df.x.apply(lambda x :x.date()) >= start) &  (df.x.apply(lambda x :x.date()) <= end) ]
#     fig.layout.yaxis.range = [in_view.y.min(), in_view.y.max()]

# fig



# py.iplot(fig, filename = 'pci', auto_open=True)
# pio.write_image(fig, 'visualization/main.svg', config{'showLink': False})

