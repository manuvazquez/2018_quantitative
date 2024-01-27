#!/usr/bin/env Rscript

# for the sake of reproducibility
set.seed(12345)

# install.packages("pROC")
library(pROC)

fixed.specificity <- 0.9
algorithm.name <- "RNN"

AUCs.output.file <- "RNN_AUC_CI.txt"
sensitivities.output.file <- "RNN_sensitivity_CI.txt"
hypothesis.tests.output.file <- "RNN_tests.txt"

markers.names <- c("CA125", "HE4", "Gly", "CA125+HE4", "CA125+Gly", "CA125+HE4+Gly")
markers.directories <- c("simple_with_time", "he4_with_time", "gly_with_time", "ca125_he4_with_time", "ca125_gly_with_time", "ca125_he4_gly_with_time")

# a vector with paths to the data files
files <- vector(mode="character", length=length(markers.directories))

# every path is built from the corresponding "marker directory"
for (i in 1:length(markers.directories))
    files[i] = paste0(markers.directories[i],'/5_folds/predictions.txt')
    
# files <- c("simple_with_time/5_folds/fold_0/predictions.txt", "he4_with_time/5_folds/fold_0/predictions.txt", "gly_with_time/5_folds/fold_0/predictions.txt")
n = length(files)

# vectors for the ROC and confidence intervals are allocated
ROC <- vector("list",n)
sensitivity.confidence.interval <- vector("list",n)

# columns are: CI lower bound, AUC estimate, CI upper bound, distance between lower bound and AUC estimate,
# distance between upper bound and AUC estimate
AUCs <- matrix(, nrow=n, ncol=5)

# same for sensitivity
sensitivities <- matrix(, nrow=n, ncol=5)

for (i in 1:n){
    
    # for the sake of convenience
    filename <- files[i]
    
    # data is read
    data <- read.csv(filename, header=FALSE, sep=",", col.names=c("actual", "prediction"))
    
    # a ROC object is computed...
    ROC[[i]] <- roc(data$actual, data$prediction)
    
    # -------
    
    # ...and from it a confidence interval obtained
    AUC.confidence.interval <- ci.auc(ROC[[i]])
    
    # confidence intervals are stored in a matrix...
    AUCs[i,1:3] <- AUC.confidence.interval[1:3]
    
    # ...along with the corresponding differences
    AUCs[i,4] <- AUC.confidence.interval[2] - AUC.confidence.interval[1]
    AUCs[i,5] <- AUC.confidence.interval[3] - AUC.confidence.interval[2]
    
    # ------
    
    sensitivity.confidence.interval <- ci.se(ROC[[i]], fixed.specificity)
    
    # confidence intervals are stored in a matrix...
    sensitivities[i,1:3] <- sensitivity.confidence.interval[1:3]
    
    # ...along with the corresponding differences
    sensitivities[i,4] <- sensitivity.confidence.interval[2] - sensitivity.confidence.interval[1]
    sensitivities[i,5] <- sensitivity.confidence.interval[3] - sensitivity.confidence.interval[2]
    
    # results are printed
    print(markers.names[i])
    print(ROC[[i]]$auc)
    print(AUC.confidence.interval)
}

# rows and columns are given names
dimnames(AUCs) <- list(markers.names, c("lowerBound", "estimate","upperBound", "lowerError", "upperError"))
dimnames(sensitivities) <- list(markers.names, c("lowerBound", "estimate","upperBound", "lowerError", "upperError"))

# memory is allocated for a matrix with the p-values of all the hypothesis tests
p.value <- matrix(, nrow=n, ncol=n)

# dimensions are named
dimnames(p.value) <- list(markers.names, markers.names)

i.p.value <- 1
# p.value.flattened <- vector(mode="numeric",length=n*(n-1)/2)
p.value.flattened <- matrix(,nrow=1,ncol=n*(n-1)/2)
p.value.test.name <- vector(mode="character",length=n*(n-1)/2)

# for every combination of markers...
for (i in 1:n){
    if (i==n) next
    for (j in (i+1):n){
        # p-value is stored in the matrix
        p.value[i,j] <- roc.test(ROC[[i]], ROC[[j]])$p.value

        p.value.flattened[i.p.value] <- p.value[i,j]
        p.value.test.name[i.p.value] <- paste0("{",paste(markers.names[i],markers.names[j],sep=" vs "),"}")

        i.p.value <- i.p.value + 1
    }
}

colnames(p.value.flattened) <- p.value.test.name

print("comparisons:")
print(p.value)

# ------- AUC

formatted.algorithm.name <- paste("\\textbf{", algorithm.name, "}",sep="")

# the header for the future table...
table.header <- paste(c("name", colnames(AUCs),"\n", formatted.algorithm.name, replicate(ncol(AUCs), "{}"),"\n"),collapse=" ")

# ...is written to the output file...
cat(table.header, file=AUCs.output.file)

# ...and finally the matrix after rounding
write.table(AUCs, AUCs.output.file,append=TRUE,quote=FALSE, sep='\t', col.names=FALSE)

cat(replicate(ncol(AUCs)+1, "{}"), file=AUCs.output.file,append=TRUE)
cat("\n", file=AUCs.output.file,append=TRUE)

# ------- sensitivity

# the header for the future table...
table.header <- paste(c("name", colnames(sensitivities),"\n", formatted.algorithm.name, replicate(ncol(sensitivities), "{}"),"\n"),collapse=" ")

# ...is written to the output file...
cat(table.header, file=sensitivities.output.file)

# ...and finally the matrix after rounding
write.table(sensitivities, sensitivities.output.file,append=TRUE,quote=FALSE, sep='\t', col.names=FALSE)

cat(replicate(ncol(AUCs)+1, "{}"), file=sensitivities.output.file,append=TRUE)
cat("\n", file=sensitivities.output.file,append=TRUE)

# ------- hypotheses tests

write.table(p.value.flattened, hypothesis.tests.output.file,quote=FALSE, sep='\t',row.names=FALSE)
