prop.methods <- c('pAC','pAUC','pAUC+US','pAUC+OS','pAUC+ST')
ref.methods <- c('SPEC','Relief','LS','EN')

# 1. Variable selection
exp <- c('proposed','ref')[2]
if(exp=='proposed'){
  methods <- prop.methods
} else if(exp=='ref'){
  methods <- ref.methods
}
res.list <- list()

for(i in 1:length(data.list)){
  print(i)
  dat <- data.list[[i]]
  
  if(exp=='proposed'){
    for(j in 1:length(methods)){
      res.list[j] <- proposed.method(data=dat, Y='Y', u=1, ntree=200, method=methods[j])
      
    }
    names(res.list) <- methods
    saveRDS(res.list, file=paste0(res.path, names(data.list[i]), '_varsec.RData'))
    
  } else if(exp=='ref'){
    for(j in 1:length(methods)){
      if(methods=='LS' | methods=='EN'){
        res.list[j] <- ref.method1(data=dat, Y='Y', family='binomial', method=methods[j]) 
        
      } else if(methods=='Relief' | methods=='SPEC'){
        res.list[j] <- ref.method2(data=dat, Y='Y', ntree=200, method=methods[j])
        
      }
    }
    names(res.list) <- methods
    saveRDS(res.list, file=paste0(res.path, names(data.list[i]), '_varsec_ref.RData'))
  }
}



# 2. Prediction (CV-AUC)
exp <- c('proposed','ref')[1]
if(exp=='proposed'){
  methods <- prop.methods
} else if(exp=='ref'){
  methods <- ref.methods
}

k <- 5 # 5-fold CV
iters <- 100
for(i in 1:length(dat.name)){
  print(i)
  auc.mat <- matrix(NA, nrow=iters, ncol=length(methods))
  colnames(auc.mat) <- methods
  
  # data processing
  dat <- data.list[[i]]
  if(exp=='proposed'){
    res <- readRDS(paste(res.path, dat.name[i], '_varsec.RData', sep=''))
    finalset.list <- list('pAC'=res$`pAC-CVER`$final.set, 'pAUC'=res$pAUC$final.set, 
                          'pAUC+US'=res$`pAUC+US`$final.set, 'pAUC+OS'=res$`pAUC+OS`$final.set, 
                          'pAUC+ST'=res$`pAUC+ST`$final.set)
    
  } else if(exp=='ref'){
    res <- readRDS(paste(res.path, dat.name[i], '_varsec_ref.RData', sep=''))
    finalset.list <- list('SPEC'=res$SPEC$final.set,'Relief'=res$Relief$final.set, 
                          'LS'=res$LS$final.set, 'EN'=res$EN$final.set)
    
  }
  finalset.list <- lapply(finalset.list, mixedsort)
  
  # CV-AUC
  for(m in 1:length(methods)){ # 1:length(finalset.list)
    print(methods[m])
    
    dat.finalset <- dat[, c(finalset.list[[methods[m]]], 'Y')] # data with selected variables only
    
    # iterations
    for(iter in 1:iters){
      print(iter)
      
      # 5-fold CV
      cv.idx <- get.kfold.splits(Y=dat.finalset$Y, k=k, seed=iter)
      cv.pred.vec <- rep(NA, k)
      for(j in 1:k){
        dat.train <- dat.finalset[c(cv.idx[[j]]$training$case, cv.idx[[j]]$training$control),]
        dat.test <- dat.finalset[c(cv.idx[[j]]$test$case, cv.idx[[j]]$test$control),]
        
        set.seed(123)
        fit.rf <- randomForest(factor(Y)~., data=dat.train, ntree=200) # ntree=100
        pred.Y <- predict(fit.rf, newdata=dat.test, type="prob")[, '1'] # prob for predicting case (0:control, 1:case)
        cv.pred.vec[j] <- fast.auc(pred.Y, dat.test$Y, reverse.sign.if.nece = FALSE) # cv-auc
      }
      
      auc.mat[iter,methods[m]] <- mean(cv.pred.vec)
    }
  }
  
  # save results
  write.table(auc.mat, file=paste0(res.path, dat.name[i], '_cvauc.txt'), col.names=TRUE, row.names=FALSE, sep=',') # set res.path
}
