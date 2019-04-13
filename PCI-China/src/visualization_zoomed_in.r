china_event <- function( vectical_position = -0.35, adj= 0.1, ep=0){
    
    out = list()
    font_size =4
    font_size2 =2.8
    rect_width = 4
    text_delta = 7
    
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
        annotate("text", x = as.Date('1978-12-18')+text_delta, y = vectical_position, hjust = 0,
                 label = "1978 reform program starts",
                 color =  ifelse(ep==5, "red" , "black"), size = ifelse(ep==5, font_size , font_size2)   )
    ))
    
    # Reform stalls
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1989-06-04'), xmax = as.Date('1989-06-04')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==6, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1989-06-04')+text_delta, y = vectical_position*0.9, hjust = 0,
                 label = "1989 Tiananmen Sq. protests",
                 color =  ifelse(ep==6, "red" , "black"), size = ifelse(ep==6, font_size , font_size2)   )
    ))
    
    # Market Economy
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('1993-11-11'), xmax = as.Date('1993-11-11')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==7, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('1993-11-11')+text_delta, y = vectical_position*0.8, hjust = 0,
                 label = "1993 reform speed-up",
                 color =  ifelse(ep==7, "red" , "black"), size = ifelse(ep==7, font_size , font_size2)   )
    ))
    
    # Harmonious Society
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2005-10-08'), xmax = as.Date('2005-10-08')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==8, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2005-10-08')+text_delta, y = vectical_position, hjust = 0,
                 label = "2005 reform slow-down",
                 color =  ifelse(ep==8, "red" , "black"), size = ifelse(ep==8, font_size , font_size2)   )
    ))
    
    # Stimulus package
    out = c(out, list(
        geom_rect(aes(xmin = as.Date('2008-11-05'), xmax = as.Date('2008-11-05')+rect_width, 
                      ymin = -Inf, ymax = Inf), alpha = 0.005 , fill = ifelse(ep==9, "mediumspringgreen" , "grey15")  ) ,
        annotate("text", x = as.Date('2008-11-05')+text_delta, y = vectical_position*0.92, hjust = 0,
                 label = "2008 stimulus package",
                 color =  ifelse(ep==9, "red" , "black"), size = ifelse(ep==9, font_size , font_size2)   )
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

prepare_data = function(model="window_5_years_quarterly", root="../"){
    data = read_csv(file.path(root,"visualization/",model,"/results.csv"))

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
common_format = function(){
    out = list()

    new = scale_x_date(
            breaks = seq.Date(as.Date("2012-01-01"),  as.Date("2019-01-01"),by="1 year"), 
            date_label = "%Y", 
            limits= as.Date(c("2012-01-01","2020-01-01")),
            expand = c(0, 0)
        ) 
    out = c(out, list(new))
}

plot_pci = function(data, event=TRUE,abs=TRUE){
    out = data %>% ggplot(aes(x=date)) 
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
        xlab("Year") + ylab("Quarterly PCI for China") +
        common_format() + theme_bw() + theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank())
        # theme(panel.background = element_rect(fill = 'grey90', colour = 'white'),
        #       panel.grid.minor = element_blank()) 
    return (out)
}

plot_F1 = function(data) {
    data %>% 
        select(c(test_F1,forecast_F1,date)) %>%
        melt(id.var='date') %>% 
        mutate(Variable = revalue(variable , c("test_F1"="Performance in testing",
                                           "forecast_F1"="Performance in \"forecasting\"") ) ) %>%
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
                 xlab("Year") + ylab("Quarterly PCI for China") +
                 scale_y_continuous(limits = c(-0.05, 0.3), breaks = seq(0,0.2,0.1), expand = c(0, 0)) +
                 common_format() +theme_bw()+
                 theme(
                       legend.position = c(0.5, 0.9),
                       legend.direction = "horizontal",
                       legend.background=element_blank(),
                       legend.key=element_blank(),
                       legend.spacing.x = unit(0.15,"cm"),
                       panel.grid.minor=element_blank(), panel.grid.major=element_blank()
                       ) +
                 scale_color_manual(values=c("blue3", "red3","black")) +
                 scale_linetype_manual(values=c("solid", "longdash","twodash"))  
    }
    out
}