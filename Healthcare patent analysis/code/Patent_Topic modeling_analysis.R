f_dir <- "C:/Users/"

NUM_TOPIC <- c(10,15,20)

analysis.type <- c("paper", "patent")
TYPE <- analysis.type[2]
rm(analysis.type)

## [1] import data
library(xlsx)
setwd(f_dir)
getwd()
final_data <- read.csv("healthcare_patents.csv",encoding='cp949') # INPUT : beta dataset
library(NLP)
library(tm)
letm_all <- read.xlsx("tokenized_result.xlsx",sheetIndex=1,encoding='UTF-8') # INPUT : lemma data

## 전처리 끝난 파일로 corpus 생성
AbsCorpus <- Corpus(VectorSource(letm_all[,2]))
rm(letm_all)


## [2] term-document matrix 만들기
#memory.limit()
AbsDTM <- DocumentTermMatrix(AbsCorpus, control = list(minWordLength = 2))
DF <- as.data.frame(inspect(AbsDTM), stringsAsFactors = FALSE)


# 검증: Each row of the input matrix needs to contain at least one non-zero entry
rowTotals <- apply(AbsDTM , 1, sum) # Find the sum of words in each Document
dtm.new   <- AbsDTM[rowTotals> 0, ] # Remove all docs without words
dim(DF)[[1]] - dim(dtm.new)[[1]]    # 이 값이 0이 아닐때, 아래 주석부분 실행 
DF <- as.data.frame(as.matrix(dtm.new))
AbsDTM <- dtm.new
rm(rowTotals)
rm(dtm.new)

write.csv(DF, "df_dfMatrix.csv") 
rm(DF)


## [3] Apply LDA algorithm
library(topicmodels)
devtools::install_github("tillbe/jsd", force=TRUE)

setwd("C:/Users/inolab/Desktop/랩실원우님/이서영원우님/result2")
start.time <- Sys.time()
for (t in 1:length(NUM_TOPIC)){
  NTopic <- NUM_TOPIC[[t]] # Topic Modeling: LDA의 토픽수 정의
  Gibbs_LDA <-LDA(AbsDTM, NTopic, method = "Gibbs", control = list(iter = 1000)) # iteration 1000 fix
  
  # 3-1. 토픽별 상위 50개 단어 목록
  Gibbs_terms <- terms(Gibbs_LDA, 50) # 각 토픽에 할당된 단어: Gibbs 추정
  write.csv(Gibbs_terms, paste("topic-term_list_",TYPE,"_t",NTopic,".csv",sep=""))
  print(paste("topic-term_list_t",NTopic,".csv",sep=""))
  rm(Gibbs_terms)
  
  # 3-2. 토픽별 문서의 확률분포
  Topic_posterior <- posterior(Gibbs_LDA)$topics # 문서의 토픽 확률
  write.csv(Topic_posterior, paste("probability_doc-topic_",TYPE,"_t",NTopic,".csv",sep=""))
  print(paste("probability_doc-topic_t",NTopic,".csv",sep=""))
  # 3-3. 토픽별 단어의 확률분포
  Term_posterior <- posterior(Gibbs_LDA)$terms # 각 토픽의 단어 출현 확률
  write.csv(Term_posterior, paste("probability_topic-term_",TYPE,"_t",NTopic,".csv",sep=""))
  print(paste("probability_topic-term_t",NTopic,".csv",sep=""))
  
  # 3-4. 토픽별 상위 20개 문서 목록
  Gibbs_topics <- topics(Gibbs_LDA, 1)
  Top20Papers <- data.frame()
  N_paper <- 20
  for (c in 1:NTopic){
    sel_idx <- order(Topic_posterior[,c],decreasing = TRUE)[1:N_paper] # 2016.8.13 수정부분
    #sel_idx <- which(Gibbs_topics == c)
    tmp_posterior <- data.frame(sel_idx, Topic_posterior[sel_idx, c])
    colnames(tmp_posterior) <- c("patent_idx", "posterior")
    tmp_posterior <- tmp_posterior[order(tmp_posterior$posterior, decreasing = TRUE),]
    tmp_topic <- rep(paste("Topic_",c, sep=""),20) 
    tmp_papers <- cbind(tmp_topic, tmp_posterior[1:20,2], final_data[tmp_posterior$patent_idx[1:20],])
    rm(sel_idx)
    rm(tmp_topic)
    rm(tmp_posterior)
    Top20Papers <- rbind(Top20Papers, tmp_papers)
    rm(tmp_papers)
  }
  write.csv(Top20Papers, paste("topic-doc_list_",TYPE,"_t",NTopic,".csv",sep=""))
  print(paste("topic-doc_list_",TYPE,"_t",NTopic,".csv",sep=""))
  rm(Top20Papers)
  
  # 3-5. 토픽별 전체확률 비중 목록
  Topic.Probability <- colSums(Topic_posterior)/dim(Topic_posterior)[[1]]
  Topic.Probability <- as.data.frame(Topic.Probability)
  write.csv(Topic.Probability, paste("topic-probability_list_",TYPE,"_t",NTopic,".csv", sep=""))
  print(paste("topic-probability_list_",TYPE,"_t",NTopic,".csv", sep=""))
  rm(Topic.Probability)
  
  # 3-6. Jensen-Shannon Divergence(Distance between Topic)
  require("jsd")
  topics <- list()
  for(i in 1:dim(Term_posterior)[[1]]){
    topics[[i]] <- prop.table(Term_posterior[i,])
  }
  DM <- matrix(,length(topics),length(topics))  # divergence matrix
  for(i in 2:length(topics)-1){
    for(j in (i+1):length(topics)){
      DM[i,j] <- sqrt(JSD(topics[[i]], topics[[j]]))
    }
  }
  rm(topics)
  write.csv(DM, paste("topic_jsd_matrix_",TYPE,"_t",NTopic,".csv",sep=""))
  print(paste("topic_jsd_matrix_",TYPE,"_t",NTopic,".csv",sep=""))
  rm(DM)
  rm(Topic_posterior)
  rm(Term_posterior)
}
end.time <- Sys.time()
end.time - start.time
getwd()