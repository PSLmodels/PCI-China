### library
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(readxl)
library(zoo)
library(extrafont)
library(lubridate)
library(dbplyr)
library(RSQLite)

# font_import(pattern="times", prompt = FALSE)

common_format = function(){
    out = list()

    new = scale_x_date(
            breaks = c(as.Date("1951-01-01"),
                       seq.Date(as.Date("1960-01-01"), as.Date("2020-01-01"), by="10 years"),
                       as.Date("2020-01-01")), 
            date_label = "%Y", 
            limits= as.Date(c("1950-01-01","2025-04-01")),
            expand = c(0, 0)
        ) 
    out = c(out, list(new))
}
prepare_data = function(model="window_5_years_quarterly", folder="figures"){
    data = read_csv(file.path(folder, model,"/results.csv"))

    data %>%
        filter(year_target>=1951) %>%
        mutate(date = as.Date(as.yearqtr(paste0(year_target, "M", mt_target+2), format = "%YM%m"), frac = 1),
               Variable = switch(model, 
                                 "window_2_years_quarterly" = "Two-year window", 
                                 "window_5_years_quarterly" = "Five-year window",  
                                 "window_10_years_quarterly" = "Ten-year window")
               )%>% 
        arrange(date)
}

figure_1 = function(input = "data/output/database.db"){
	con <- dbConnect(RSQLite::SQLite(), input)
	res <- dbGetQuery(con, "SELECT year, quarter, count(1) n, avg(frontpage) frontpage  FROM main group by year, quarter")
	dbDisconnect(con)
	
	data = res %>% 
		as_tibble() %>%
		mutate(date = as.Date( as.yearqtr(paste0(year, " Q", quarter)) ) ) %>%
		select(date, frontpage, n) %>%
		mutate(n = n/ 1000)
	
	hh = 3 

    # data$Qdate = as.Date(data$Qdate)
    # data$frontpage = as.numeric(data$frontpage)
    # data$id = as.numeric(data$id/1000)

    sf = 1/15
    sf_C = 0

    fig = ggplot(data, aes(x = date)) +
        geom_line(aes(y = frontpage, colour = "Fraction of front-page articles")) +
        geom_line(aes(y = n*sf + sf_C , colour = "Number of articles")) +
        scale_x_date(
            breaks = c(as.Date("1946-01-01"),
                       seq.Date(as.Date("1950-01-01"), as.Date("2020-01-01"), by="10 years"),
                       as.Date("2020-01-01")),
            date_label = "%Y",
            limits = as.Date(c("1945-01-01","2025-04-01")),
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

    fig
}


figure_2 = function(input= "data/input/china_capital_outflows_SAFE.xlsx", models_folder="figures"){

    ######
    outflow_data = read_excel(input) %>%
        mutate(date = as.Date(as.yearqtr(paste0(year, "M", month), format = "%YM%m"), frac = 1)) %>%
        select(-year, -month) %>%
        arrange(date)

    outflow_data = rename(outflow_data, eo="EO, 4Q rolling sum ($bn)")
    outflow_data = outflow_data %>% mutate(eo = eo/1000)

    data_5y = prepare_data("window_5_years_quarterly", models_folder ) %>% mutate(shade_1= 0.7*( (lubridate::year(date) > 2005) + 0) )

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
        geom_text( aes( x=as.Date("1986-01-01"), y = 0.85), label = "Dual-track reform",family = "Times New Roman") +
        geom_text( aes( x=as.Date("2012-05-01"), y = 0.85), label = "Harmonious society",family = "Times New Roman")
        # geom_text( aes( x=as.Date("2010-01-01"), y = 0.65), label = "Accelerating \ncapital flight",family = "Times", color="red")
    fig
}


china_event <- function( vectical_position = -0.35, adj= 0.1, ep=0){
    
    out = list()
    font_size =4
    font_size2 =2.8
    rect_width = 40
    text_delta = 60

    # # Regime established.
    # out = c(out, list(
    #     geom_rect(aes(xmin = as.Date('1949-10-01'), xmax = as.Date('1949-10-01')+rect_width, 
    #                   ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==1, "mediumspringgreen" , "grey20")  ) ,
    #     annotate("text", x = as.Date('1949-10-01')+text_delta, y = vectical_position,
    #              label = "1949: Regime est.",
    #              color =  ifelse(ep==1, "red" , "black"), size = ifelse(ep==1, font_size , font_size2)   )
    # ))
    
    # First Five-Year Plan
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1953-01-01'), xmax = as.Date('1953-01-01')+rect_width,
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==1, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1953-01-01')+text_delta, y = vectical_position, hjust = 0,
                 label = "1953 first Five-Year Plan",
                 color =  ifelse(ep==1, "red" , "black"), size = ifelse(ep==1, font_size , font_size2)   )
    ))
    
    # Great Leap Forward
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1958-05-16'), xmax = as.Date('1958-05-16')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==2, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1958-05-16')+text_delta, y = vectical_position*0.9, hjust = 0,
                 label = "1958 Great Leap Forward",
                 color =  ifelse(ep==2, "red" , "black"), size = ifelse(ep==2, font_size , font_size2)   )
    ))
    
    # Cultural Revolution
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1966-05-16'), xmax = as.Date('1966-05-16')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==3, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1966-05-16')+text_delta, y = vectical_position*0.8, hjust = 0,
                 label = "1966 Cultural Revolution",
                 color =  ifelse(ep==3, "red" , "black"), size = ifelse(ep==3, font_size , font_size2)   )
    ))
    
    # Hua takes over.
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1976-09-09'), xmax = as.Date('1976-09-09')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==4, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1976-09-09')+text_delta, y = vectical_position*0.7, hjust = 0,
                 label = "1976 Hua takes over",
                 color =  ifelse(ep==4, "red" , "black"), size = ifelse(ep==4, font_size , font_size2)   )
    ))
    
    # Reform era starts.
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1978-12-18'), xmax = as.Date('1978-12-18')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==5, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1978-12-18')+text_delta, y = vectical_position*0.6, hjust = 0,
                 label = "1978 reform program starts",
                 color =  ifelse(ep==5, "red" , "black"), size = ifelse(ep==5, font_size , font_size2)   )
    ))
    
    # Reform stalls
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1989-06-04'), xmax = as.Date('1989-06-04')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==6, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1989-06-04')+text_delta, y = vectical_position, hjust = 0,
                 label = "1989 Tiananmen Sq. protests",
                 color =  ifelse(ep==6, "red" , "black"), size = ifelse(ep==6, font_size , font_size2)   )
    ))
    
    # Market Economy
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1993-11-11'), xmax = as.Date('1993-11-11')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==7, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1993-11-11')+text_delta, y = vectical_position*0.9, hjust = 0,
                 label = "1993 reform speed-up",
                 color =  ifelse(ep==7, "red" , "black"), size = ifelse(ep==7, font_size , font_size2)   )
    ))
    
    # SARS
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2002-11-16'), xmax = as.Date('2002-11-16')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==8, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2002-11-16')+text_delta, y = vectical_position, hjust = 0,
                 label = "2002 SARS outbreak",
                 color =  ifelse(ep==8, "red" , "black"), size = ifelse(ep==8, font_size , font_size2)   )
    ))
    
    # Harmonious Society
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2005-10-08'), xmax = as.Date('2005-10-08')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==9, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2005-10-08')+text_delta, y = vectical_position*0.8, hjust = 0,
                 label = "2005 reform slowdown",
                 color =  ifelse(ep==9, "red" , "black"), size = ifelse(ep==9, font_size , font_size2)   )
    ))
    
    # Stimulus package
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2008-11-05'), xmax = as.Date('2008-11-05')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==10, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2008-11-05')+text_delta, y = vectical_position*0.6, hjust = 0,
                 label = "2008 stimulus package",
                 color =  ifelse(ep==10, "red" , "black"), size = ifelse(ep==10, font_size , font_size2)   )
    ))
    
    # Renewed reforms
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2013-04-22'), xmax = as.Date('2013-04-22')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==11, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2013-04-22')+text_delta, y = vectical_position, hjust = 0,
                 label = "2013 revive Maoism",
                 color =  ifelse(ep==11, "red" , "black"), size = ifelse(ep==11, font_size , font_size2)   )
    ))
    
    # Renewed reforms
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2013-09-11'), xmax = as.Date('2013-09-11')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==12, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2013-09-11')+text_delta, y = vectical_position*0.9, hjust = 0,
                 label = "2013 renew reform",
                 color =  ifelse(ep==12, "red" , "black"), size = ifelse(ep==12, font_size , font_size2)   )
    ))
    
    # Supply-side reform
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2015-11-10'), xmax = as.Date('2015-11-10')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==13, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2015-11-10')+text_delta, y = vectical_position*0.78, hjust = 0,
                 label = "2015 supply-side\nreform",
                 color =  ifelse(ep==13, "red" , "black"), size = ifelse(ep==13, font_size , font_size2)   )
    ))
    
    # COVID-19
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2019-12-27'), xmax = as.Date('2019-12-27')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==14, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2019-12-27')+text_delta*4, y = vectical_position*0.4, hjust = 0, angle = 90,
                 label = "COVID-19 outbreak",
                 color =  ifelse(ep==14, "red" , "black"), size = ifelse(ep==14, font_size , font_size2)   )
    ))
    
    out
}

plot_pci = function(data, event=TRUE,abs=TRUE){
    out = data %>% ggplot(aes(x=date)) 
    if (abs){
        out = out + geom_line(aes(y=pci ), colour="blue3", show.legend = FALSE) +
            scale_y_continuous(limits = c(-0.05, 0.45), breaks = seq(0,0.4,0.2), expand = c(0, 0)) 

    } 
    else {
        out = out + geom_line(aes(y=diff), colour="blue3", show.legend = FALSE) +
            scale_y_continuous(limits = c(-0.25, 0.45), breaks = seq(-0.2,0.4,0.1), expand = c(0, 0)) 
    }
    if (event){
        out = out + china_event(vectical_position=0.4, adj= 0.05)
    }

    out = out + 
        geom_hline(yintercept = 0, linetype = 2, color = "black")  +
        xlab("Year") + ylab("Quarterly PCI-China") +
        common_format() + theme_bw() + theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank())
        # theme(panel.background = element_rect(fill = 'grey90', colour = 'white'),
        #       panel.grid.minor = element_blank()) 
    return (out)
}

plot_F1 = function(data) {
    data %>% 
        select(c(test_F1,forecast_F1,date)) %>%
        melt(id.var='date') %>% 
        mutate(Variable = recode(variable , 
        						 test_F1=" Performance in testing    ",
                                 forecast_F1=" Performance in \"forecasting\"") )  %>%
        ggplot( aes(x=date, y=value, col=Variable)) +
        geom_line() +
        xlab("Year") + ylab("Classification performance") +
        scale_y_continuous(limits = c(0.3, 1), breaks = seq(0.4,1,0.2), expand = c(0, 0)) +
        common_format() + theme_bw()+
        theme(
              legend.position = c(0.5, 0.9),
              legend.direction = "horizontal",
              legend.background=element_blank(),
              legend.key=element_blank(),
              legend.spacing.x = unit(0.15,"cm"),
              legend.title=element_blank(),
              panel.grid.minor=element_blank(), panel.grid.major=element_blank()
              ) +
        scale_color_manual(values=c("orange2", "purple2"))  
}


plot_compare = function(data1, data2, data3, type=c("F1","PCI")){

    type = match.arg(type)
    data = bind_rows(data1,data2,data3)

    if (type == "F1"){
        out = 
        data %>% ggplot( aes(x=date, y=test_F1, col=Variable, linetype=Variable)) +
                 geom_line() +
                 xlab("Year") + ylab("Classification performance") +
                 scale_y_continuous(limits = c(0.5, 0.9), breaks = seq(0.5,0.9,0.2), expand = c(0, 0)) +
                 common_format() + theme_bw()+
                 theme(
                       legend.position = c(0.5, 0.9),
                       legend.direction = "horizontal",
                       legend.background=element_blank(),
                       legend.key=element_blank(),
                       legend.title=element_blank(),
                       legend.spacing.x = unit(0.15,"cm"),
                       panel.grid.minor=element_blank(), panel.grid.major=element_blank()
                       ) +
                 scale_color_manual(values=c("orange2", "black","blue")) +
                 scale_linetype_manual(values=c("solid", "longdash","twodash"))  
    }
    if (type == "PCI"){
        out = 
        data %>% ggplot( aes(x=date, y=pci, col=Variable, linetype=Variable)) +
                 geom_line() +
                 geom_hline(yintercept = 0, linetype = 2, color = "black")  +
                 xlab("Year") + ylab("Quarterly PCI-China") +
                 scale_y_continuous(limits = c(-0.05, 0.45), breaks = seq(0,0.4,0.2), expand = c(0, 0)) +
                 common_format() +theme_bw()+
                 theme(
                       legend.position = c(0.5, 0.9),
                       legend.direction = "horizontal",
                       legend.background=element_blank(),
                       legend.key=element_blank(),
                       legend.title=element_blank(),
                       legend.spacing.x = unit(0.15,"cm"),
                       panel.grid.minor=element_blank(), panel.grid.major=element_blank()
                       ) +
                 scale_color_manual(values=c("blue3", "red3","black")) +
                 scale_linetype_manual(values=c("solid", "longdash","twodash"))  
    }
    out
}




plot_pci_since_2012 = function(data, event=TRUE,abs=TRUE){
    common_format = function(){
        out = list()

        new = scale_x_date(
                breaks = seq.Date(as.Date("2012-01-01"),  as.Date("2022-01-01"),by="1 year"), 
                date_label = "%Y", 
                limits= as.Date(c("2011-10-01","2025-04-01")),
                expand = c(0, 0)
            ) 
        out = c(out, list(new))
    }

    china_event <- function( vectical_position = -0.35, adj= 0.1, ep=0){
        
        out = list()
        font_size =4
        font_size2 =2.8
        rect_width = 4
        text_delta = 7

        out = c(out, list(
            geom_rect(aes(xmin = as.Date('1953-01-01'), xmax = as.Date('1953-01-01')+rect_width,
                          ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==1, "mediumspringgreen" , "grey15")  ) ,
            annotate("text", x = as.Date('1953-01-01')+text_delta, y = vectical_position, hjust = 0,
                     label = "1953 first Five-Year Plan",
                     color =  ifelse(ep==1, "red" , "black"), size = ifelse(ep==1, font_size , font_size2)   )
        ))
      
        # Renewed reforms
        out = c(out, list(
            geom_rect(aes(xmin = as.Date('2013-04-22'), xmax = as.Date('2013-04-22')+rect_width, 
                          ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==10, "mediumspringgreen" , "grey15")  ) ,
            annotate("text", x = as.Date('2013-04-22')+text_delta, y = vectical_position*1, hjust = 0,
                     label = "2013 revive Maoism",
                     color =  ifelse(ep==10, "red" , "black"), size = ifelse(ep==10, font_size , font_size2)   )
        ))
        
        # Renewed reforms
        out = c(out, list(
            geom_rect(aes(xmin = as.Date('2013-09-11'), xmax = as.Date('2013-09-11')+rect_width, 
                          ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==11, "mediumspringgreen" , "grey15")  ) ,
            annotate("text", x = as.Date('2013-09-11')+text_delta, y = vectical_position*0.9, hjust = 0,
                     label = "2013 renew reform program",
                     color =  ifelse(ep==11, "red" , "black"), size = ifelse(ep==11, font_size , font_size2)   )
        ))
        
        # Supply-side reform
        out = c(out, list(
            geom_rect(aes(xmin = as.Date('2015-11-10'), xmax = as.Date('2015-11-10')+rect_width, 
                          ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==12, "mediumspringgreen" , "grey15")  ) ,
            annotate("text", x = as.Date('2015-11-10')+text_delta, y = vectical_position*0.8, hjust = 0,
                     label = "2015 supply-side structural reform",
                     color =  ifelse(ep==12, "red" , "black"), size = ifelse(ep==12, font_size , font_size2)   )
        ))
        out
    }


    out = data  %>% ggplot(aes(x=date)) 
    if (abs){
        out = out + geom_line(aes(y=pci ), colour="blue3", show.legend = FALSE) +
            scale_y_continuous(limits = c(-0.05, 0.3), breaks = seq(0,0.2,0.1), expand = c(0, 0)) 

    } 
    else {
        out = out + geom_line(aes(y=diff), colour="blue3", show.legend = FALSE) +
            scale_y_continuous(limits = c(-0.25, 0.45), breaks = seq(-0.2,0.4,0.1), expand = c(0, 0)) 
    }
    if (event){
        out = out + china_event(vectical_position=0.25, adj= 0.05)
    }

    out = out + 
        geom_hline(yintercept = 0, linetype = 2, color = "black")  +
        xlab("Year") + ylab("Quarterly PCI-China") +
        common_format() + theme_bw() + theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank())
        # theme(panel.background = element_rect(fill = 'grey90', colour = 'white'),
        #       panel.grid.minor = element_blank()) 
    return (out)
}


gen_summary_statistics = function(input = "data/output/database.db", output="figures/Summary statistics.csv"){
  con <- dbConnect(RSQLite::SQLite(), input)
  res <- dbGetQuery(con, "SELECT year, quarter, count(1) number_of_articles, 
    avg(frontpage) frontpage , avg(page1to3) page1to3, avg(n_articles_that_day) avg_n_articles_per_day, avg(n_pages_that_day) avg_n_pages_per_day, avg(n_frontpage_articles_that_day) avg_n_frontpage_articles_per_day, 
    avg(title_len) avg_n_of_word_seg_in_title,  min(title_len) min_n_of_word_seg_in_title, max(title_len) max_n_of_word_seg_in_title,
    avg(body_len) avg_n_of_word_seg_in_body,  min(body_len) min_n_of_word_seg_in_body, max(body_len) max_n_of_word_seg_in_body
    FROM main group by year, quarter")
  write_csv(res, output)
  dbDisconnect(con)
}




plot_pci_2005_reform_slow_down = function(data, event=TRUE,abs=TRUE){
	common_format = function(){
		out = list()
		
		new = scale_x_date(
			breaks = seq.Date(as.Date("1997-01-01"),  as.Date("2008-03-01"),by="1 year"), 
			date_label = "%Y", 
			limits= as.Date(c("1997-01-01","2008-03-01")),
			expand = c(0, 0)
		) 
		out = c(out, list(new))
	}
	
	china_event <- function( vectical_position = -0.35, adj= 0.1, ep=0){
		
		out = list()
		font_size = 4.3
		font_size2 =2.8
		rect_width = 4
		text_delta = 7
		
		out = c(out, list(
			geom_rect(aes(xmin = as.Date('2005-10-08'), xmax = as.Date('2005-10-08')+rect_width, 
						  ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==8, "mediumspringgreen" , "grey15")  ) ,
			annotate("text", x = as.Date('2005-10-08')+text_delta, y = vectical_position, hjust = -0.05,
					 label = "2005 reform slowdown",
					 color =  ifelse(ep==8, "red" , "black"), size = font_size   )
		))
		
		
		
		out
	}
	
	
	out = data  %>% ggplot(aes(x=date)) 
	if (abs){
		out = out + geom_line(aes(y=pci ), colour="blue3", show.legend = FALSE) +
			scale_y_continuous(limits = c(-0.05, 0.4), breaks = seq(0,0.5,0.1), expand = c(0, 0)) 
		
	} else {
		out = out + geom_line(aes(y=diff), colour="blue3", show.legend = FALSE) +
			scale_y_continuous(limits = c(-0.25, 0.5), breaks = seq(-0.2,0.4,0.1), expand = c(0, 0)) 
	}
	if (event){
		out = out + china_event(vectical_position=0.25, adj= 0.05)
	}
	
	out = out + 
		geom_hline(yintercept = 0, linetype = 2, color = "black")  +
		xlab("Year") + ylab("Quarterly PCI-China") +
		common_format() + theme_bw() + theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank())
	# theme(panel.background = element_rect(fill = 'grey90', colour = 'white'),
	#       panel.grid.minor = element_blank()) 
	return (out)
}

