#source("~/Library/CloudStorage/Box-Box/PRRS nomen/Analysis 2-Full VDL/trimtree.R")
#library(missForest)

library(ape)
library(seqinr)
library(adegenet)
library(caret)
library(randomForest)
library(Biostrings)
library(stringr)

#load("model.rf10v.new.lin.Rdata")
load(url("https://github.com/kvanderwaal/prrsv2_classification/raw/main/model.rf10v.new.lin.11.2023.Rdata"))
xg_lin <- xg_fit
load(url("https://github.com/kvanderwaal/prrsv2_classification/raw/main/model.Rdata"))

subset.align <- function(align,n){
  align.temp.nam <- align$nam[align$nam %in% n]
  align.temp.seq <- align$seq[align$nam %in% n]
  align.sub <- seqinr::as.alignment(nb=length(align.temp.nam),nam=align.temp.nam,seq=align.temp.seq)
  return(align.sub)
}

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

reference_sequence <- DNAString("ATGTTGGAGAAATGCTTGACCGCGGGCTGTTACTCGCAATTGCTTTCTTTGTGGTGTATCGTGCCGTTCTGTTTTGCTGTGCTCGTCAACGCCAGCAACGACAGCAGCTCCCATCTACAGCTGATTTACAACTTGACGCTATGTGAGCTGAATGGCACAGATTGGCTAGCTAACAAATTTGATTGGGCAGTGGAGAGTTTTGTCATCTTTCCCGTTTTGACTCACATTGTCTCCTATGGTGCCCTCACTACTAGCCATTTCCTTGACACAGTCGCTTTAGTCACTGTGTCTACCGCCGGGTTTGTTCACGGGCGGTATGTCCTAAGTAGCATCTACGCGGTCTGTGCCCTGGCTGCGTTGACTTGCTTCGTCATTAGGTTTGCAAAGAATTGCATGTCCTGGCGCTACGCGTGTACCAGATATACCAACTTTCTTCTGGACACTAAGGGCAGACTCTATCGTTGGCGGTCGCCTGTCATCATAGAGAAAAGGGGCAAAGTTGAGGTCGAAGGTCATCTGATCGACCTCAAAAGAGTTGTGCTTGATGGTTCCGTGGCAACCCCTATAACCAGAGTTTCAGCGGAACAATGGGGTCGTCCTTAG")

# Function to trim sequences to 603 bases
trim_to_603 <- function(seq) {
  if (nchar(seq) > 603) {
    return(substr(seq, start = 1, stop = 603))
  } else {
    return(seq)
  }
}

# Function to remove non-nucleotide characters from sequences
remove_non_nucleotides <- function(seq) {
  gsub("[^ACGTacgt]", "", seq)
}

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




#function runs with RF object and an alingnment. This function is identical to the main make.predict fun, and is used internally to assign sub-lineage to undetermined sequences
make.predict.lin <- function(m=xg_lin,al.new=al.new)    {
  #realign to reference
  long_names <- which(sapply(al.new$seq, nchar) != 603)
  
  right_names <- al.new$nam[which(sapply(al.new$seq, nchar) == 603)]
  right_seq <- al.new$seq[which(sapply(al.new$seq, nchar) == 603)]
  #right_subset <- as.alignment(nb=length(right_names),nam=right_names,seq=right_seq)
  right_subset <- c(nam=right_names,seq=right_seq)
  
  # Initialize an empty list to store aligned sequences
  aligned_sequences_list <- list()
  if(length(long_names)>0){
    for (i in long_names) {
      #print(i)
      # Remove gaps from the sequence
      aligned_seq <- remove_non_nucleotides(al.new$seq[[i]])
      
      # Convert the sequence to a DNAString object
      al.new.1 <- DNAString(aligned_seq)
      
      # Define the substitution matrix
      mat <- nucleotideSubstitutionMatrix(match = 1, mismatch = -3, baseOnly = TRUE)
      
      # Perform pairwise alignment
      localAlign <- pairwiseAlignment(reference_sequence, al.new.1, substitutionMatrix = mat,
                                      gapOpening = 25, gapExtension = 15, type = "global-local")
      
      # Extract the aligned sequence
      aligned_seq <- as.character(aligned(localAlign@subject))
      
      # if greater than 603, need to remove the insertion
      if(nchar(aligned_seq)>603){
        insert <- as.data.frame(indel(localAlign)@deletion[[1]])
        insert <- insert[order(-insert$start),]
        for(j in 1:nrow(insert)){
          insert.j <- seq(insert$start,insert$end,by=1)
          aligned_seq <- paste(unlist(strsplit(aligned_seq,""))[-insert.j], collapse = "")
        }
      }
      
      # Convert the aligned sequence to a DNAStringSet object
      aligned_seq_set <- DNAStringSet(aligned_seq)
      names(aligned_seq_set) <- al.new$nam[i]  # Set the name of the sequence
      
      # Save the aligned sequences to a list or an appropriate structure
      aligned_sequences_list[[i]] <- aligned_seq_set
      
    }
    compiled_aligned_sequences <- do.call(c, aligned_sequences_list)
    
    # Save the compiled aligned sequences to a single FASTA file
    temp_dir <- tempdir()  # Create a temporary directory
    fasta_file_path <- file.path(temp_dir, "compiled_aligned_sequences.fasta")  # Define the path for the FASTA file
    compiled_aligned_sequences.1 <- unlist(DNAStringSetList(compiled_aligned_sequences))
    writeXStringSet(compiled_aligned_sequences.1, file = fasta_file_path, format = "fasta")
    
    # Read the compiled aligned sequences back into an alignment object
    aligned_trim_seq <- read.alignment(file = fasta_file_path, format = "fasta")
    
    # Exclude sequences that were aligned from the original al.new and then add the aligned sequences
    seq_to_exclude <- which(nchar(al.new$seq) != 603)
    nam_to_exclude <- al.new[["nam"]][seq_to_exclude]
    al.new.nam <- al.new$nam[!(al.new$nam %in% nam_to_exclude)]
    al.new.seq <- al.new$seq[!(al.new$nam %in% nam_to_exclude)]
    al.new.sub <- seqinr::as.alignment(nb=length(al.new.nam),nam=al.new.nam,seq=al.new.seq)
    
    al.new <- seqinr::as.alignment(nb=length(c(al.new.nam,aligned_trim_seq[["nam"]])),
                                   nam=c(al.new.nam,aligned_trim_seq[["nam"]]),
                                   seq=c(al.new.seq,aligned_trim_seq$seq))  # Add the aligned sequences to the original al.new
    
    # Exclude and print names of non-conforming seqeuences
    seq_to_exclude <- which(nchar(al.new$seq) != 603)
    nam_to_exclude <- al.new[["nam"]][seq_to_exclude]
    al.new.nam <- al.new$nam[!(al.new$nam %in% nam_to_exclude)]
    al.new.seq <- al.new$seq[!(al.new$nam %in% nam_to_exclude)]
    al.new <- seqinr::as.alignment(nb=length(al.new.nam),nam=al.new.nam,seq=al.new.seq)
  }
  amb <-unlist(lapply(al.new$seq,str_count,pattern="-"))
  name.list <- al.new$nam
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
  
  p.out <- data.frame(SequenceName=(rownames(d.rf)),
                      assign.final=NA,
                      assign.top=p,
                      prob.top=round(apply(probs,1,max),3),
                      assign.2 = assign.sec[2,],
                      prob.2 = as.numeric(assign.sec[1,]),
                      assign.3 = assign.third[2,],
                      prob.3 = as.numeric(assign.third[1,])
  )
  p.out$assign.2 <- ifelse(p.out$prob.2==0,NA,p.out$assign.2)
  p.out$assign.3 <- ifelse(p.out$prob.3==0,NA,p.out$assign.3)
  
  p.out$assign.final <- ifelse(p.out$prob.top>=.25 & p.out$prob.top/p.out$prob.2>=2,as.character(p.out$assign.top),"undetermined")
 
  p.out$num.gaps.amb <- amb
  p.out$SequenceName <- name.list
  
  return(p.out[,c(1:6,9)])
}


#function runs with a RF object and an alingnment
make.predict <- function(m=xg_fit,al.new=al.new)    {
  #realign to reference
  long_names <- which(sapply(al.new$seq, nchar) != 603)
  
  right_names <- al.new$nam[which(sapply(al.new$seq, nchar) == 603)]
  right_seq <- al.new$seq[which(sapply(al.new$seq, nchar) == 603)]
  #right_subset <- as.alignment(nb=length(right_names),nam=right_names,seq=right_seq)
  right_subset <- c(nam=right_names,seq=right_seq)
  
  # Initialize an empty list to store aligned sequences
  aligned_sequences_list <- list()
  if(length(long_names)>0){
    for (i in long_names) {
      #print(i)
      # Remove gaps from the sequence
      aligned_seq <- remove_non_nucleotides(al.new$seq[[i]])
      
      # Convert the sequence to a DNAString object
      al.new.1 <- DNAString(aligned_seq)
      
      # Define the substitution matrix
      mat <- nucleotideSubstitutionMatrix(match = 1, mismatch = -3, baseOnly = TRUE)
      
      # Perform pairwise alignment
      localAlign <- pairwiseAlignment(reference_sequence, al.new.1, substitutionMatrix = mat,
                                      gapOpening = 25, gapExtension = 15, type = "global-local")
      
      # Extract the aligned sequence
      aligned_seq <- as.character(aligned(localAlign@subject))
      
      # if greater than 603, need to remove the insertion
      if(nchar(aligned_seq)>603){
        insert <- as.data.frame(indel(localAlign)@deletion[[1]])
        insert <- insert[order(-insert$start),]
        for(j in 1:nrow(insert)){
          insert.j <- seq(insert$start,insert$end,by=1)
          aligned_seq <- paste(unlist(strsplit(aligned_seq,""))[-insert.j], collapse = "")
        }
        }
      
      # Convert the aligned sequence to a DNAStringSet object
      aligned_seq_set <- DNAStringSet(aligned_seq)
      names(aligned_seq_set) <- al.new$nam[i]  # Set the name of the sequence
      
      # Save the aligned sequences to a list or an appropriate structure
      aligned_sequences_list[[i]] <- aligned_seq_set
      
    }
    compiled_aligned_sequences <- do.call(c, aligned_sequences_list)
    
    # Save the compiled aligned sequences to a single FASTA file
    temp_dir <- tempdir()  # Create a temporary directory
    fasta_file_path <- file.path(temp_dir, "compiled_aligned_sequences.fasta")  # Define the path for the FASTA file
    compiled_aligned_sequences.1 <- unlist(DNAStringSetList(compiled_aligned_sequences))
    writeXStringSet(compiled_aligned_sequences.1, file = fasta_file_path, format = "fasta")
    
    # Read the compiled aligned sequences back into an alignment object
   aligned_trim_seq <- read.alignment(file = fasta_file_path, format = "fasta")
    
    # Exclude sequences that were aligned from the original al.new and then add the aligned sequences
    seq_to_exclude <- which(nchar(al.new$seq) != 603)
    nam_to_exclude <- al.new[["nam"]][seq_to_exclude]
    al.new.nam <- al.new$nam[!(al.new$nam %in% nam_to_exclude)]
    al.new.seq <- al.new$seq[!(al.new$nam %in% nam_to_exclude)]
    al.new.sub <- seqinr::as.alignment(nb=length(al.new.nam),nam=al.new.nam,seq=al.new.seq)
    
    al.new <- seqinr::as.alignment(nb=length(c(al.new.nam,aligned_trim_seq[["nam"]])),
                             nam=c(al.new.nam,aligned_trim_seq[["nam"]]),
                             seq=c(al.new.seq,aligned_trim_seq$seq))  # Add the aligned sequences to the original al.new
    
    # Exclude and print names of non-conforming seqeuences
    seq_to_exclude <- which(nchar(al.new$seq) != 603)
    nam_to_exclude <- al.new[["nam"]][seq_to_exclude]
    al.new.nam <- al.new$nam[!(al.new$nam %in% nam_to_exclude)]
    al.new.seq <- al.new$seq[!(al.new$nam %in% nam_to_exclude)]
    al.new <- seqinr::as.alignment(nb=length(al.new.nam),nam=al.new.nam,seq=al.new.seq)
    }
  amb <-unlist(lapply(al.new$seq,str_count,pattern="-"))
  name.list <- al.new$nam
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
  
  p.out <- data.frame(SequenceName=(rownames(d.rf)),
                      assign.final=NA,
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
  p.out$ratios <- p.out$prob.top/p.out$prob.2
  
  p.out$num.gaps.amb <- amb
  
  p.out$SequenceName <- name.list
  al.und <- subset.align(al.new,al.new$nam[al.new$nam %in% p.out$SequenceName[p.out$assign.final=="undetermined" | p.out$ratios<2]])
  
  if(al.und$nb >0 ){
    lin.pred <- make.predict.lin(al.new=al.und)
    lin.pred$assign.final2 <- ifelse(lin.pred$assign.final=="undetermined","undetermined",paste(lin.pred$assign.final,"unclassified",sep="-"))
    p.out$assign.final[p.out$SequenceName %in% lin.pred$SequenceName] <- lin.pred$assign.final2[match(p.out$SequenceName[p.out$SequenceName %in% lin.pred$SequenceName],lin.pred$SequenceName )]
  }
  
  
  return(p.out[,c(1:6,10)])
}
