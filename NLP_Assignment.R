library(tm)
library(ggplot2)
library(tidytext)
library(gutenbergr)
library(dplyr)
library(tidyr)
library(qdap)
library(tibble)
library(RWeka)
library(lubridate)
library(lexicon)
library(lubridate)
library(stringr)
library(radarchart)
library(sentimentr)
library(wordcloud)

setwd("C:/Users/PMK/Desktop/Desktop School/Data 902/emails")

#######################################################################################################################
# 1. Twitter feed text analysis (minimum 2000 tweets)/Other data sources
#I'm using the Sarah Palin E-Mails
set.seed(17)

files <- sample(list.files(), 1000)

filelists <- readtext::readtext(files)

#Pre-Processing
clean_corpus <- function(cleaned_corpus){
  removeURL <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
  cleaned_corpus <- tm_map(cleaned_corpus, removeURL)
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  # available stopwords
  # stopwords::stopwords()
  custom_stop_words <- c("msnbc", "dot","com","sent","pro","publica","can","weve","draft","will","address","state","govpalinyahoo","searchable","govsarahyahoo","govepalinyahoo","device","date","posted","and","from","(gov)","re:","the","have","to:","cc:","etc","web","kelly","analytics","alaska","redacted","www","re","gov","governor","subject","email","sent","sarah","palin","subject:","have","from:","://","message","fsearchable","original","w")
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words)
  # cleaned_corpus <- tm_map(cleaned_corpus, stemDocument,language = "english")
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}

filelists$text <- iconv(filelists$text, from = "UTF-8", to = "ASCII", sub = "")
files_corpus <- VCorpus(VectorSource(filelists$text))
cleaned_files_corpus <- clean_corpus(files_corpus)
TDM_files <- TermDocumentMatrix(cleaned_files_corpus)
files_tidy <- tidy(TDM_files)

########################################################################################################################
# 2. TF word clouds
# 
# Unigram
# Bigram
# Trigram

TDM_emails_m <- as.matrix(TDM_files)

#Unicode
#Count the number of times each term is used, and to make it easier to view (if needed), the second line
#Arranges the terms in descending order
term_frequency <- rowSums(TDM_emails_m)
term_frequency <- sort(term_frequency,dec=TRUE)

word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=1000,colors=brewer.pal(8, "Paired"))

#############################
#Bigram
bi_tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=2,max=2))

bigram_tdm <- TermDocumentMatrix(cleaned_files_corpus,control = list(tokenize=bi_tokenizer))
bigram_matrix <- as.matrix(bigram_tdm)

bi_term_frequency <- rowSums(bigram_matrix)
# Sort term_frequency in descending order
bi_term_frequency_sorted <- sort(bi_term_frequency,dec=TRUE)

bi_word_freqs <- data.frame(term = names(bi_term_frequency_sorted), num = bi_term_frequency)

# Create a wordcloud for the values in word_freqs
wordcloud(bi_word_freqs$term, bi_word_freqs$num,min.freq=5,max.words=1000, random.order = FALSE,colors=brewer.pal(8, "Paired"))

#############################
#Trigrams
tritokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=5,max=5))

trigram_tdm <- TermDocumentMatrix(cleaned_files_corpus,control = list(tokenize=tritokenizer))
trigram_matrix <- as.matrix(trigram_tdm)

tri_term_frequency <- rowSums(trigram_matrix)
# Sort term_frequency in descending order
tri_term_frequency <- sort(tri_term_frequency,dec=TRUE)

tri_word_freqs <- data.frame(term = names(tri_term_frequency), num = tri_term_frequency)

# Create a wordcloud for the values in word_freqs
wordcloud(tri_word_freqs$term, tri_word_freqs$num,min.freq=5,max.words=1000, random.order = FALSE,colors=brewer.pal(8, "Paired"))

########################################################################################################################
# 3. TF-IDF word cloud
tfidf_emails <- TermDocumentMatrix(cleaned_files_corpus,control=list(weighting=weightTfIdf))
tfidf_emails_m <- as.matrix(tfidf_emails)

# Term Frequency
files_terms_frequency <- rowSums(tfidf_emails_m)
# Sort term_frequency in descending order
term_files_frequency <- sort(files_terms_frequency,dec=TRUE)
############Word Cloud

# Create word_freqs
word_freqs <- data.frame(term = names(term_files_frequency), num = term_files_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=1000,colors=brewer.pal(8, "Paired"))

########################################################################################################################
# 4. Sentiment analysis (any one lexicon)
bing_lex <- get_sentiments("bing")
bing_lex_pos_neg <- bing_lex[bing_lex$sentiment %in% c("positive","negative"),]
files_bing_lex <- inner_join(files_tidy, bing_lex_pos_neg, by = c("term" = "word"))
files_bing_lex$sentiment_n <- ifelse(files_bing_lex$sentiment=="negative", -1, 1)
files_bing_lex$sentiment_value <- files_bing_lex$sentiment_n * files_bing_lex$count
bing_aggdata <- aggregate(files_bing_lex$sentiment_value, list(index = files_bing_lex$document), sum)
bing_aggdata$index <- as.numeric(bing_aggdata$index)
colnames(bing_aggdata) <- c("index","bing_score")

tidy_sentiment_data <- gather(bing_aggdata, sentiment_dict, sentiment_score, -index)
tidy_sentiment_data[is.na(tidy_sentiment_data)] <- 0

ggplot(data = tidy_sentiment_data,
       aes(x=index,y=sentiment_score,fill=sentiment_dict))+
  geom_bar(stat="identity") + facet_grid(sentiment_dict~.)+theme_bw() + 
  theme(legend.position = "none")+ggtitle("E-Mail Sentiment")

########################################################################################################################
# 5. Comparison/Contrast word clouds based on sentiment
indices <- filelists

indices$index <- bing_aggdata$index

allset <- merge(indices, tidy_sentiment_data, by = 'index')

allset$index <- NULL
allset$doc_id <- NULL
allset$sentiment_dict <- NULL

positiveemails <- allset[allset$sentiment_score > 0,]
negativeemails <- allset[allset$sentiment_score < 0,]

positiveemails$sentiment_score <- NULL
negativeemails$sentiment_score <- NULL

negemails <- paste(unlist(gettext(positiveemails)), collapse =" ")
posemails <- paste(unlist(gettext(negativeemails)), collapse =" ")

allemails <- c(posemails,negemails)

emails_V <- VCorpus(VectorSource(allemails))

cleaned_emails <- clean_corpus(emails_V)

theemail_tdm <- TermDocumentMatrix(cleaned_emails)
colnames(theemail_tdm) <- c("Positive","Negative")

emails_m <- as.matrix(theemail_tdm)

commonality.cloud(emails_m,colors=brewer.pal(8, "Dark2"),max.words = 1000)
comparison.cloud(emails_m,colors=brewer.pal(8, "Dark2"),max.words = 200)

########################################################################################################################
#6. Emotional analysis (any one lexicon)

nrc_lex <- get_sentiments("nrc")
story_nrc <- inner_join(files_tidy, nrc_lex, by = c("term" = "word"))
story_nrc_noposneg <- story_nrc[!(story_nrc$sentiment %in% c("positive","negative")),]
aggdata <- aggregate(story_nrc_noposneg$count, list(index = story_nrc_noposneg$sentiment), sum)

#Easily the two biggest sentiments in the graph are Trust and Anticipation.
chartJSRadar(aggdata)




