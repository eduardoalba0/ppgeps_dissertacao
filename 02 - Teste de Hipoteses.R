rm(list=ls(all=TRUE))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(forecast)
library(ggplot2) 
library(dplyr)
library(lubridate)
library(reshape2)
library(feasts)
library(fpp3)
library(urca)
library(trend)
library(randtests)

Sys.setlocale("LC_ALL", "pt_BR")
options(encoding = "UTF-8")
set.seed(1)

datasets <-list.files(path = "./dados/series", pattern= "\\.csv$", full.names=TRUE)

teste_hipoteses <- data.frame(
  'CAMPUS' = character(),
  'REGIÃO' = character(),
  'TESTE' = character(),
  'VALOR DO TESTE' = numeric(),
  'VALOR CRÍTICO' = numeric(),
  'CONCLUSÃO' = character(),
  stringsAsFactors = FALSE
)

for (dataset in datasets) {
  data<-read.table(file=dataset, header=TRUE, sep=";", dec=".")
  
  campus = data$CAMPUS[1]
  regiao = data$REGIÃO[1]
  
  data$PERIODO <- as.Date(data$PERIODO, format = "%Y-%m-%d")
 
  ano <- as.numeric(format(data$PERIODO[1], "%Y"))
  mes <- as.numeric(format(data$PERIODO[1], "%m"))
  
  dts<-ts(data$CONSUMO, start = c(ano, mes), frequency = 12)
  
  #################################KPSS######################################
  
  args(ur.kpss)
  
  kpss_test<-ur.kpss(dts,type="tau",lags="short")
  kpss_resultado = kpss_test@teststat
  kpss_significancia = kpss_test@cval[1,2]
  
  
  if (kpss_resultado > kpss_significancia) {
    kpss_conclusao = "Possui raiz unitária - Não Estacionário"
  } else { 
    kpss_conclusao = "Não possui raiz unitária - Estacionário"
  }
  
  
  nova_linha <- data.frame(
    'CAMPUS' = campus,
    'REGIÃO' = regiao,
    'TESTE' = "KPSS",
    'VALOR DO TESTE' = kpss_resultado,
    'VALOR CRÍTICO' = kpss_significancia,
    'CONCLUSÃO' = kpss_conclusao,
    stringsAsFactors = FALSE
  )
  teste_hipoteses <- rbind(teste_hipoteses, nova_linha)
  
  ############################Man-Kendall###################################
  
  mankendall_test<-mk.test(dts)
  mk_pvalue = mankendall_test$p.value
  
  if (mk_pvalue < 0.05) {
    mk_conclusao = "Não possui tendência"
  } else { 
    mk_conclusao = "Possui tendência"
  }
  
  nova_linha <- data.frame(
    'CAMPUS' = campus,
    'REGIÃO' = regiao,
    'TESTE' = "Man Kenndall",
    'VALOR DO TESTE' = mk_pvalue,
    'VALOR CRÍTICO' = 0.05,
    'CONCLUSÃO' = mk_conclusao,
    stringsAsFactors = FALSE
  )
  teste_hipoteses <- rbind(teste_hipoteses, nova_linha)
  
  ############################Kruskal-Wallis#############################
  
  kw_test<-kruskal.test(data$CONSUMO ~ data$ORDEM)
  kw_pvalue = kw_test$p.value
  print(kw_pvalue)
  
  if (kw_pvalue < 0.05) {
    kw_conclusao = "Possui Sazonalidade"
  } else {
    kw_conclusao = "Não possui Sazonalidade"
  }
  
  nova_linha <- data.frame(
    'CAMPUS' = campus,
    'REGIÃO' = regiao,
    'TESTE' = "Kruskal Wallis",
    'VALOR DO TESTE' = kw_pvalue,
    'VALOR CRÍTICO' = 0.05,
    'CONCLUSÃO' = kw_conclusao,
    stringsAsFactors = FALSE
  )
  teste_hipoteses <- rbind(teste_hipoteses, nova_linha)
  
}

print(teste_hipoteses)
write.table(teste_hipoteses, "./dados/hipoteses.csv",  row.names = FALSE, sep = ";", fileEncoding = "UTF-8", col.names = TRUE, quote = TRUE)