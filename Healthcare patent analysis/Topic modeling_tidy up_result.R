library(xlsx)

f_dir <- "C:/Users/"

setwd(f_dir)

topic_num<-'20'

file_name <- paste0("topic-term_list_patent_t",topic_num,'.csv')

file_name2<-paste0('probability_topic-term_patent_t',topic_num,'.csv')

topic_list <- read.csv(file_name,encoding='UTF-8')
topic_list<-topic_list[,-1]

term_rate <- read.csv(file_name2, encoding='UTF-8')
term_rate<-term_rate[,-1]


# 토픽에 속한 단어의 비중열 추가 -> rate1= 1번 토픽에 해당하는 단어들의 비율

for (i in 1:as.numeric(topic_num)){
  a<-term_rate[i,]
  b<-sort(a, decreasing = T)
  b<-b[1:50]
  c<-paste0('rate',as.character(i))
  topic_list[c]<-as.numeric(b)
}


write.xlsx(topic_list, "result_20.xlsx") # 저장파일이름 바꿔가면서 저장하기.


# topic_num을 바꾸어가며, 위 코드 실행