checkpoint:::checkpoint(
    snapshotDate = utils::packageDescription("RevoUtils")$MRANDate , 
    R.version = "3.5.1")

### Set path
.PATH = getwd()
.ROOT = file.path(.PATH)
.OUTPUT =  file.path(.ROOT,"visualization")    

### library
library(readr)
library(plyr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(zoo)
library(plotly)
library(reshape2)
source("src/visualization_zoomed_in.r")

hh = 4

######

data_5y = prepare_data("window_5_years_quarterly", .ROOT )
write_csv(data_5y, file.path(.OUTPUT, "window_5_years_quarterly","pci.csv"))

data_10y = prepare_data("window_10_years_quarterly", .ROOT)
data_2y = prepare_data("window_2_years_quarterly", .ROOT)

ggsave(file.path(.OUTPUT,"window_5_years_quarterly","pci_with_events_5y_since_2012.png"),
       plot = plot_pci(data_5y,event=TRUE,abs=TRUE), width=hh*2.4 ,height=1.5*hh)
