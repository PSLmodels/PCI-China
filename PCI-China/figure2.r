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
source("src/visualization.r")
library(readxl)
library(data.table)
library(extrafont)
font_import()

hh = 4

######
outflow_data = read_excel("data/input/china_capital_outflows_SAFE.xlsx") %>%
    mutate(date = as.Date(as.yearqtr(paste0(year, "M", month), format = "%YM%m"), frac = 1)) %>%
    select(-year, -month) %>%
    arrange(date)
outflow_data = rename(outflow_data, eo="EO, 4Q rolling sum ($bn)")
outflow_data = outflow_data %>% mutate(eo = eo/1000)

data_5y = prepare_data("window_5_years_quarterly", .ROOT ) %>% mutate(shade_1= 0.7*( (lubridate::year(date) > 2005) + 0) )

n = 10000
date_seq = seq( as.Date("2005-10-08") , as.Date("2019-01-01"), length.out= n+1)

data_tile = rbind(    
                data.frame(
                    xmin = head(date_seq,-1),
                    xmax = tail(date_seq,-1),
                    ymin = 0.9,
                    ymax = 0.5,
                    alpha = seq(from=0.25,to=0,length.out=n)
                ),
                data.frame(
                    xmin = as.Date("1978-12-18"),
                    xmax = as.Date("1993-11-11"),
                    ymin = 0.9,
                    ymax = 0.5,
                    alpha = 0.15
                )
            )


fig = ggplot(data_5y) +
    geom_rect(data= data_tile, aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,alpha=alpha), fill="gray" ) + 
    guides(alpha = FALSE, guide_legend(title="New Legend Title"))+
    
    geom_line(aes(x=date, y=test_F1, col=Variable, linetype=Variable)) +
    xlab("Year") + ylab("Model performance (F1 score)") +
    scale_y_continuous(limits = c(0.5, 0.9), breaks = seq(0.5,0.9,0.2), expand = c(0, 0)) +
    common_format() + theme_bw()+
    theme(
        legend.position = "none",
        legend.direction = "horizontal",
        legend.background=element_blank(),
        legend.key=element_blank(),
        panel.grid.minor=element_blank(), panel.grid.major=element_blank()
    ) +
    scale_color_manual(values=c("orange2")) +
    scale_linetype_manual(values=c("solid")) +
    # geom_line(data=outflow_data, aes(y = eo*1.1 + 0.8, x=date), size = 0.5,color ="red") +
    geom_text( aes( x=as.Date("1986-01-01"), y = 0.85), label = "Dual-track reform",family = "Times") +
    geom_text( aes( x=as.Date("2012-05-01"), y = 0.85), label = "Harmonious society",family = "Times")
    # geom_text( aes( x=as.Date("2010-01-01"), y = 0.65), label = "Accelerating \ncapital flight",family = "Times", color="red")
fig 


ggsave(file.path(.OUTPUT,"Joint","5_years_test_F1_w_shaded.png")) 

