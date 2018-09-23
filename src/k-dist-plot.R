library(dbscan)

X_New = read.csv("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment3/X_normalized.csv", sep=",") 

X_New <- as.matrix(X_New)
dim(X_New)

for (i in seq(3, 21, 2)) {  
  pdf(paste("k-dist-graph",i,".pdf",sep = "")) 
  kNNdistplot(X_New, k=i)
  dev.off()
}


