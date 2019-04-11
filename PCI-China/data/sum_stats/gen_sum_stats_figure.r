checkpoint:::checkpoint(
    snapshotDate = utils::packageDescription("RevoUtils")$MRANDate , 
    R.version = "3.5.1")

### library
library(readr)
# library(plyr)
library(dplyr)
library(ggplot2)
# library(tidyr)
# library(zoo)
# library(plotly)
library(reshape2)

hh = 3.5

######

data = read_csv("sum_stats.csv")

data$Qdate = as.Date(data$Qdate)
data$frontpage = as.numeric(data$frontpage)
data$id = as.numeric(data$id/1000)

sf = 1/15
sf_C = 0

plot_sum_stats = ggplot(data, aes(x = Qdate)) +
    geom_line(aes(y = frontpage, colour = "Fraction of front-page articles")) +
    geom_line(aes(y = id*sf + sf_C , colour = "Number of articles")) +
    scale_x_date(
        breaks = c(seq.Date(as.Date("1946-01-01"), as.Date("2018-12-31"), by="8 years")),
        date_label = "%Y",
        limits = as.Date(c("1945-01-01","2020-12-31")),
        expand = c(0, 0)
    ) +
    xlab("Year") +
    scale_y_continuous(
        "Fraction of front-page articles (quarterly)",
        sec.axis = sec_axis(~((.-sf_C)/sf), name = "Number of articles (quarterly, thousands)"),
        limit = c(0,0.85)
    ) +
    theme_bw()+
    theme(
        legend.position = c(0.2, 0.9),
        legend.background=element_blank(),
        legend.key=element_blank(),
        legend.title=element_blank(),
        panel.grid.minor=element_blank(), panel.grid.major=element_blank()
    ) +
    scale_color_manual(values=c("blue3", "red3"))

ggsave("sum_stats.png", plot = plot_sum_stats, width=hh*2.4 ,height=1.5*hh)
