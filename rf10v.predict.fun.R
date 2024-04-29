#source("~/Library/CloudStorage/Box-Box/PRRS nomen/Analysis 2-Full VDL/trimtree.R")
#library(missForest)

library(ape)
library(seqinr)
library(adegenet)
library(caret)
library(randomForest)

#load("model.rf10v.new.lin.Rdata")
#load("~/Library/CloudStorage/Box-Box/PRRS nomen/Lineage RF/Lineage RF - 07.2023/model.rf10v.new.lin.11.2023.Rdata")
rep.row<-function(x,n){
  matrix(rep(x,each=n),nrow=n)
}

calc_mode <- function(x){
  # List the distinct / unique values
  distinct_values <- unique(x)
  # Count the occurrence of each distinct value
  distinct_tabulate <- tabulate(match(x, distinct_values))
  # Return the value with the highest occurrence
  distinct_values[which.max(distinct_tabulate)]
}
# Not neede any more 2024-01-11
# #replace NA with mode in a vector
# na.to.mode <- function(x){
# 
#   ifelse(is.na(x)==T,calc_mode(x),x)
# }


#al10 <- read.alignment("~/Library/CloudStorage/Box-Box/PRRS nomen/Analysis_1:2- Clean_seq/Analysis 1-2019-21/sequences.fasta",format="fasta")

#attr <- read.csv("~/Library/CloudStorage/Box-Box/PRRS nomen/Analysis_1:2- Clean_seq/Analysis 1-2019-21/iqtree full/attr2.csv")

#al.train.test <- subset.align(al10,n=attr$strain)

#add some dummy data to alignment to ensure that all possible alleles are represented
dummy.seqs <- data.frame(seqs=strrep("a",603))
dummy.seqs <- rbind(dummy.seqs,data.frame(seqs=strrep("t",603)))
dummy.seqs <- rbind(dummy.seqs,data.frame(seqs=strrep("g",603)))
dummy.seqs <- rbind(dummy.seqs,data.frame(seqs=strrep("c",603)))
dummy.seqs <- rbind(dummy.seqs,data.frame(seqs=strrep("-",603))) #account for gaps in sequence
#align.dummy <- as.alignment(nb=(al.train.test$nb)+nrow(dummy.seqs),nam=c(paste("dummy",1:5),al.train.test$nam),seq=c(dummy.seqs$seqs,al.train.test$seq))

align.dummy <- seqinr::as.alignment(nb=nrow(dummy.seqs),nam=c(paste("dummy",1:5)),seq=c(dummy.seqs$seqs))

#d <- DNAbin2genind(al.sub.train.bin)
#d.test <- DNAbin2genind(al.sub.test.bin)
d.all <- DNAbin2genind(as.DNAbin.alignment(align.dummy))
d.all <- d.all[-(1:5)]




#function runs with a dapc function and an alingnment
make.predict <- function(m=xg_fit,al.new=al.new)    {
  
  #add dummy dable to al.new so that it ahas same column structure
  align.dummy.new <- seqinr::as.alignment(nb=(al.new$nb)+nrow(dummy.seqs),
                                  nam=c(paste("dummy",1:5),al.new$nam),
                                  seq=c(dummy.seqs$seqs,al.new$seq))
  
  
  #format data against original d.all to ensure that it has all same columns
      d.c <- DNAbin2genind(as.DNAbin.alignment(align.dummy.new),polyThres=1/(align.dummy.new$nb))
      # print(align.dummy.new)
     
    
   d.rf <- genind2df(d.c)
   #ensure read as factor
  
   
   
   names(d.rf) <- paste("p",names(d.rf),sep=".")
   
   #remove columns that aren't used in training
   d.rf <- d.rf[,colnames(d.rf) %in% names(m$trainingData)]
   
   modes <- as.character((sapply(m$trainingData[2:ncol(m$trainingData)],calc_mode)))
   names(modes) <- names(m$trainingData[2:ncol(m$trainingData)])
  # modes <- modes[-1]
   
   #THIS LOOPS IS TURNIGN THING INTO CHARACTERS.  MNOVE LAPPLY TO BELOW?
   for(f in 1:ncol(d.rf)){
     #levs <- levels(d.rf[,f])
     d.rf[,f] <- ifelse(is.na(d.rf[,f])==T,modes[f],d.rf[,f])
     d.rf[,f]
   }
     
   d.rf[,1:ncol(d.rf)] <- lapply(d.rf[,1:ncol(d.rf)],factor)
   lapply(d.rf[,1:ncol(d.rf)],levels)
   
   d.rf <- d.rf[-(1:5),]#remove dummies
   #convert NA to mode#convert NA calc_modeto mode
   #d.rf[,1:ncol(d.rf)] <- sapply(d.rf[,1:ncol(d.rf)],na.to.mode) 
   
  
   # not needed any more!! 2024-01-11
   # na.to.x <- function(vec){
   #   #vec <- factor(vec,levels=c(levels(vec),"x"))
   #   ifelse(is.na(vec)==T,"x",vec)
   # }
   #d.rf <- sapply(d.rf,na.to.x)
   
   for (f in 2:length(names(m$trainingData))) {
     nam <- names(m$trainingData)[f]
     #find levels in new data that are not in training 2024-01-11
     drop <- levels(d.rf[,nam])[levels(d.rf[,nam]) %in% levels(m$trainingData[,nam])==F]
     for(i in 1:nrow(d.rf)) { 
        if(d.rf[i,nam] %in% drop){
         d.rf[i,nam] <- ifelse(d.rf[i,nam] %in% drop,modes[nam],d.rf[i,nam])
         #d.rf[,nam] <- droplevels(d.rf[,nam],exclude=drop)
        }
     }
       levs <- levels(d.rf[,nam]) #get bases for new data 2024-01-11 
       mode <- (calc_mode(m$trainingData[,nam]))#get position modes
       #apply modified calc_mode function to fill NAs
       #d.rf[,nam] <- (d.rf[,nam])
   #    #d.rf[,nam] <- ifelse(is.na(d.rf[,nam])==T ,mode,d.rf[,nam])
       #d.rf[,nam]<- factor(d.rf[,nam],labels=levs)
       
       d.rf[,nam] <- factor(d.rf[,nam],levels=levels(m$trainingData[,nam]))
    }
    
     
   #replace NA (produced by dropping elvels with missforest 
   #Not needed any more 11/01/2024
   #d.rf.imp <- missForest(d.rf)$ximp
   #replace NA with mode
   #d.rf.imp <- sapply(d.rf,na.to.mode) 
   
    p <- predict(m,newdata=d.rf)
    probs <- predict(m,newdata=d.rf,type="prob")
    
    #get second most likely 
    get.alts <- function(x,alt=2){
      ind <- (sort(x)[length(x) - (alt-1)]) 
      id <-names(ind)
      return(c(ind,id))
    }
    
    assign.sec <- apply(probs,1,get.alts)
    assign.third <- apply(probs,1,get.alts,alt=3)
    
    p.out <- data.frame(strain=(rownames(d.rf)),
                        assign.top=p,
                        prob.top=round(apply(probs,1,max),3),
                        assign.2 = assign.sec[2,],
                        prob.2 = as.numeric(assign.sec[1,]),
                        assign.3 = assign.third[2,],
                        prob.3 = as.numeric(assign.third[1,])
                        )
    p.out$assign.2 <- ifelse(p.out$prob.2==0,NA,p.out$assign.2)
    p.out$assign.3 <- ifelse(p.out$prob.3==0,NA,p.out$assign.3)
    p.out$assign.final <- ifelse(p.out$prob.top<.25,"undetermined",as.character(p.out$assign.top))
    return(p.out)
}
