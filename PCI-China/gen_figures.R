checkpoint:::checkpoint(
    snapshotDate = utils::packageDescription("RevoUtils")$MRANDate , 
    R.version = "3.5.3")

.OUTPUT = "figures"
source("src/figures.r")
library(R.devices)


fig1 = figure_1(input = "data/output/database.db")
suppressGraphics( ggsave("figures/others/figure_1.png", plot = fig1, width=3*2.4 ,height=3*1.5) )

fig2 = figure_2(input= "data/input/china_capital_outflows_SAFE.xlsx", models_folder="figures")

suppressGraphics( ggsave(filename="figure_2.png", plot=fig2,  width=4*2.74 ,height=4*1.5, device="png", path = "figures/others/") ) 


## Prepare data
data_5y = prepare_data("window_5_years_quarterly", folder = "figures" )
write_csv(data_5y, file.path(.OUTPUT,"pci.csv"))

data_10y = prepare_data("window_10_years_quarterly", folder = "figures")
data_2y = prepare_data("window_2_years_quarterly", folder = "figures")

## Figure 3 
hh = 4 
suppressGraphics( ggsave(file.path("figures","others","figure 3.png"),
	   plot = plot_pci_since_2012(data_5y,event=TRUE,abs=TRUE), width=hh*2.4 ,height=1.5*hh))



### 5 years rolling windows 

suppressGraphics( ggsave(file.path(.OUTPUT,"window_5_years_quarterly","diff_wo_events_5y.png"), plot = plot_pci(data_5y,event=FALSE,abs=FALSE) , width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_5_years_quarterly","pci_wo_events_5y.png"), plot = plot_pci(data_5y,event=FALSE,abs=TRUE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_5_years_quarterly","diff_with_events_5y.png"), plot = plot_pci(data_5y,event=TRUE,abs=FALSE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_5_years_quarterly","pci_with_events_5y.png"), plot = plot_pci(data_5y,event=TRUE,abs=TRUE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_5_years_quarterly","F1_two_curves_5y.png"), plot = plot_F1(data_5y), width=hh*2.4 ,height=1.5*hh))


### 10 years rolling windows 

suppressGraphics( ggsave(file.path(.OUTPUT,"window_10_years_quarterly", "diff_wo_events_10y.png"), plot = plot_pci(data_10y,event=FALSE,abs=FALSE) , width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_10_years_quarterly", "pci_wo_events_10y.png"), plot = plot_pci(data_10y,event=FALSE,abs=TRUE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_10_years_quarterly", "diff_with_events_10y.png"), plot = plot_pci(data_10y,event=TRUE,abs=FALSE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_10_years_quarterly", "pci_with_events_10y.png"), plot = plot_pci(data_10y,event=TRUE,abs=TRUE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_10_years_quarterly", "F1_two_curves_10y.png"), plot = plot_F1(data_10y), width=hh*2.4 ,height=1.5*hh))


### 2 years rolling windows 

suppressGraphics( ggsave(file.path(.OUTPUT,"window_2_years_quarterly", "diff_wo_events_2y.png"), plot = plot_pci(data_2y,event=FALSE,abs=FALSE) , width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_2_years_quarterly", "pci_wo_events_2y.png"), plot = plot_pci(data_2y,event=FALSE,abs=TRUE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_2_years_quarterly", "diff_with_events_2y.png"), plot = plot_pci(data_2y,event=TRUE,abs=FALSE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_2_years_quarterly", "pci_with_events_2y.png"), plot = plot_pci(data_2y,event=TRUE,abs=TRUE), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"window_2_years_quarterly", "F1_two_curves_2y.png"), plot = plot_F1(data_2y), width=hh*2.4 ,height=1.5*hh))



### Compare years
suppressGraphics( ggsave(file.path(.OUTPUT,"Others","figure_4.png"), plot = plot_compare(data_5y,data_10y,data_2y,"F1"), width=hh*2.4 ,height=1.5*hh))
suppressGraphics( ggsave(file.path(.OUTPUT,"Others","figure_5.png"), plot = plot_compare(data_5y,data_10y,data_2y,"PCI"), width=hh*2.4 ,height=1.5*hh))



gen_summary_statistics()
