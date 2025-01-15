# Packages
library(randomForest); library(tidyverse); library(themis); library(gtools); library(glmnet)
library(FSelectorRcpp); library(Rdimtools); library(smotefamily); library(BayesFactor)

# Simulation data
sim.imbalanced=function(n1, n0, seed) {
  set.seed(seed)
  
  N <- n1+n0
  strong <- matrix(rnorm(N*5, mean=0, sd=1), nrow=N, ncol=5) # 5 predictors; strong effect
  strong[1:n1,] <- strong[1:n1,] + 1
  moderate <- matrix(rnorm(N*5, mean=0, sd = 1), nrow=N, ncol=5) # 5 predictors; moderate effect
  moderate[1:n1,] <- moderate[1:n1,] + 0.75
  weak <- matrix(rnorm(N*5, mean=0, sd=1), nrow=N, ncol=5) # 5 predictors; week effect
  weak[1:n1,] <- weak[1:n1,] + 0.5
  noise <- matrix(rnorm(N*15, mean=0, sd=1), nrow=N, ncol=15) # 15 noise predictors
  
  # outcome
  Y <- c(rep(1, each=n1), rep(0, each=n0))
  dat <- cbind(Y, strong, moderate, weak, noise)
  colnames(dat) <- c('Y', paste('X', 1:30, sep=''))
  dat <- data.frame(dat)
  dat
}

# OOB-prediction in RF
get.oob.pred <- function(fit, data, Y, perm=FALSE, perm.X=NULL, metric=c('OOB-ER','OOB-AUC')){
  inbag <- fit$inbag
  ntree <- fit$ntree
  
  oob.pred.vec <- rep(NA, length=ntree)
  for(i in 1:ntree){
    oob.idx <- inbag[,i] == 0
    oob.sample <- data[oob.idx,]
    oob.sample.X <- oob.sample %>% select(-Y)
    oob.sample.Y <- oob.sample$Y %>% as.numeric
    
    # OOB prediction
    if(perm==FALSE){
      if(metric=='OOB-ER'){
        # OOB-error
        pred.Y <- predict(fit, newdata=oob.sample.X, type="response", predict.all=TRUE)$individual[,i] %>% as.numeric
        oob.pred.vec[i] <- sum(oob.sample.Y != pred.Y) / length(oob.sample.Y)
      } else if(metric=='OOB-AUC'){
        # OOB-AUC
        pred.Y <- predict(fit, newdata=oob.sample.X, type="prob", predict.all=TRUE)$individual[,i] %>% as.numeric # prob should 0 or 1 because terminal nodes are pure
        oob.pred.vec[i] <- fast.auc(pred.Y, oob.sample.Y, reverse.sign.if.nece = FALSE)
      } 
    }
    
    # OOB prediction after permuting a predictor  
    if(perm==TRUE){
      temp.X <- oob.sample.X
      set.seed(123) # set the same random seed for permutation
      temp.X[,perm.X] <- oob.sample.X[sample(nrow(oob.sample.X)), perm.X]
      oob.sample.perm.X <- temp.X
      
      if(metric=='OOB-ER'){
        # OOB-ER
        pred.Y <- predict(fit, newdata=oob.sample.perm.X, type="response", predict.all=TRUE)$individual[,i] %>% as.numeric
        oob.pred.vec[i] <- sum(oob.sample.Y != pred.Y) / length(oob.sample.Y)
      } else if(metric=='OOB-AUC'){
        # OOB-AUC
        pred.Y <- predict(fit, newdata=oob.sample.perm.X, type="prob", predict.all=TRUE)$individual[,i] %>% as.numeric
        oob.pred.vec[i] <- fast.auc(pred.Y, oob.sample.Y, reverse.sign.if.nece = FALSE)
      }
    }
  }
  
  return(oob.pred.vec)
}

# RF permutation variable importance
get.vim <- function(data, Y, ntree=200, method=c('pAC','pAUC','pAUC+US','pAUC+OS','pAUC+ST')){
  p <- ncol(data)-1
  oob.pred.vec <- rep(NA, p)
  oob.pred.se.vec <- rep(NA, p)
  names(oob.pred.vec) <- names(oob.pred.se.vec) <- paste('X', 1:p, sep='')
  
  if(method=='pAC'){
    # OOB-AC
    set.seed(123)
    fit.rf <- randomForest(factor(Y)~., data=data, ntree=ntree, keep.inbag=TRUE, keep.forest=TRUE, importance=TRUE)
    oob.er <- get.oob.pred(fit=fit.rf, data=data, Y='Y', perm=FALSE, metric='OOB-ER') 
    oob.ac <- 1-oob.er # accuracy = 1-error
    
    for(k in 1:p){
      #print(k)
      oob.er.perm <- get.oob.pred(fit=fit.rf, data=data, Y=Y, perm=TRUE, perm.X=k, metric='OOB-ER') # OOB-ER after permuting a predictor
      oob.ac.perm <- 1-oob.er.perm
      vi <- oob.ac - oob.ac.perm
      oob.pred.vec[k] <- mean(vi[!is.nan(vi)]) 
      oob.pred.se.vec[k] <- sd(vi[!is.nan(vi)])/sqrt(length(vi[!is.nan(vi)]))
    }
    
  } else if(method=='pAUC'){
    # OOB-AUC
    set.seed(123)
    fit.rf <- randomForest(factor(Y)~., data=data, ntree=ntree, keep.inbag=TRUE, keep.forest=TRUE, importance=TRUE)
    oob.pred <- get.oob.pred(fit=fit.rf, data=data, Y=Y, perm=FALSE, metric='OOB-AUC')
    
    for(k in 1:p){
      #print(k)
      oob.perm.pred <- get.oob.pred(fit=fit.rf, data=data, Y=Y, perm=TRUE, perm.X=k, metric='OOB-AUC') # OOB-AUC after permutation
      vi <- oob.pred - oob.perm.pred
      oob.pred.vec[k] <- mean(vi[!is.nan(vi)]) 
      oob.pred.se.vec[k] <- sd(vi[!is.nan(vi)])/sqrt(length(vi[!is.nan(vi)]))
    }
    
  } else if(method == 'pAUC+US'){
    # OOB-AUC with under-sampling
    y.ratio <- table(data$Y)
    minor.class <- as.numeric(names(y.ratio)[which.min(y.ratio)]) # under-sampling
    
    set.seed(123)
    fit.rf <- randomForest(factor(Y)~., data=data, ntree=ntree, keep.inbag=TRUE, keep.forest=TRUE,
                           sampsize=c(sum(data$Y==minor.class),sum(data$Y==minor.class))) # no need to set weights, sampsize automatically computes weights and draws a specific number of samples for each class
    oob.pred <- get.oob.pred(fit=fit.rf, data=data, Y=Y, perm=FALSE, metric='OOB-AUC') # OOB-AUC
    
    for(k in 1:p){
      #print(k)
      oob.perm.pred <- get.oob.pred(fit=fit.rf, data=data, Y=Y, perm=TRUE, perm.X=k, metric='OOB-AUC') # OOB-AUC after permutation
      diff.auc <- oob.pred - oob.perm.pred
      oob.pred.vec[k] <- mean(diff.auc[!is.nan(diff.auc)])
      oob.pred.se.vec[k] <- sd(diff.auc[!is.nan(diff.auc)])/sqrt(length(diff.auc[!is.nan(diff.auc)]))
    }
    
  } else if(method == 'pAUC+OS'){
    # OOB-AUC with over-sampling
    y.ratio <- table(data$Y)
    minor.class <- as.numeric(names(y.ratio)[which.min(y.ratio)])
    major.class <- as.numeric(names(y.ratio)[which.max(y.ratio)])
    if(minor.class == major.class){ # dealing with balanced data
      minor.class <- 1
      major.class <- 0
    }
    n.major.class <- sum(data$Y == major.class)
    set.seed(123)
    over.idx <- sample(which(data$Y == minor.class), n.major.class, replace=TRUE) # over-sampling minority class
    data.over <- data[c(over.idx, which(data$Y == major.class)),]
    rownames(data.over) <- 1:nrow(data.over)
    
    set.seed(123)
    fit.rf <- randomForest(factor(Y)~., data=data.over, ntree=ntree, keep.inbag=TRUE, keep.forest=TRUE)
    oob.pred <- get.oob.pred(fit=fit.rf, data=data.over, Y=Y, perm=FALSE, metric='OOB-AUC') # OOB-AUC
    
    for(k in 1:p){
      #print(k)
      oob.perm.pred <- get.oob.pred(fit=fit.rf, data=data.over, Y=Y, perm=TRUE, perm.X=k, metric='OOB-AUC') # OOB-AUC after permutation
      diff.auc <- oob.pred - oob.perm.pred
      oob.pred.vec[k] <- mean(diff.auc[!is.nan(diff.auc)])
      oob.pred.se.vec[k] <- sd(diff.auc[!is.nan(diff.auc)])/sqrt(length(diff.auc[!is.nan(diff.auc)]))
    }
    
  } else if(method=='pAUC+ST'){
    # OOB-AUC with SMOTE sampling
    data$Y <- as.factor(data$Y)
    if(sum(data$Y==1) <=2){
      data.st <- smotenc(data, var="Y", k=1) # with rare cases
    } else if(sum(data$Y==1) <=4){
      data.st <- smotenc(data, var="Y", k=3) # with rare cases
    } else if(sum(data$Y==1) >4){
      data.st <- smotenc(data, var="Y", k=5) # default
    }
    data.st$Y <- as.numeric(as.vector(data.st$Y))
    
    set.seed(123)
    fit.rf <- randomForest(factor(Y)~., data=data.st, ntree=ntree, keep.inbag=TRUE, keep.forest=TRUE)
    oob.pred <- get.oob.pred(fit=fit.rf, data=data.st, Y=Y, perm=FALSE, metric='OOB-AUC') # OOB-AUC
    
    for(k in 1:p){
      #print(k)
      oob.perm.pred <- get.oob.pred(fit=fit.rf, data=data.st, Y=Y, perm=TRUE, perm.X=k, metric='OOB-AUC') # OOB-AUC after permutation
      diff.auc <- oob.pred - oob.perm.pred
      oob.pred.vec[k] <- mean(diff.auc[!is.nan(diff.auc)])
      oob.pred.se.vec[k] <- sd(diff.auc[!is.nan(diff.auc)])/sqrt(length(diff.auc[!is.nan(diff.auc)]))
    }
  
  }
  
  return(list('oob.pred'=oob.pred.vec, 'oob.pred.se'=oob.pred.se.vec))
}

# K-fold cross validation
get.kfold.splits <- function (Y, k, seed){
  set.seed(seed)
  n0 = sum(Y==0)
  n1 = sum(Y==1)
  training.subsets = list()
  test.subsets = list()
  tmp1 = sample(which(Y==1), replace=FALSE)
  tmp0 = sample(which(Y==0), replace=FALSE)
  splits = list()
  for (ki in 1:k) {
    splits[[ki]] = list(training = list(case = tmp1[(1:n1)%%k != ki - 1], control = tmp0[(1:n0)%%k != ki - 1]), test = list(case = tmp1[(1:n1)%%k == ki - 1], control = tmp0[(1:n0)%%k == ki - 1]))
  }
  splits
}

# Proposed method
proposed.method <- function(data, Y, u=1, ntree, method=c('pAC','pAUC','pAUC+US','pAUC+OS','pAUC+ST')){
  
  ## Search stage
  vim <- get.vim(data=data, Y=Y, ntree=ntree, method=method)
  order.idx <- order(vim$oob.pred, decreasing=TRUE)
  vim.sorted <- vim$oob.pred[order.idx]
  vim.se.sorted <- vim$oob.pred.se[order.idx]
  
  # Iterate fitting RF with smaller number of features (remove a set of features whose VI is within upper limit of the worst VI among the remaining predictors)
  p <- ncol(data)-1
  candidate.vars <- c(p)
  current.p <- p
  while(current.p > 1){
    select.p <- unname(which( (vim.sorted - u*vim.se.sorted) <= (vim.sorted[current.p] + u*vim.se.sorted[current.p]) )[1]) # mean < upper bound
    if(select.p != 1){
      next.p <- select.p - 1
    } else if(select.p == 1){
      next.p <- 1
    }
    candidate.vars <- c(candidate.vars, next.p)
    current.p <- next.p
    
    # p=1 include or not
    if((vim.sorted - u*vim.se.sorted)[1] <= (vim.sorted[candidate.vars[length(candidate.vars)-1]] + u*vim.se.sorted[candidate.vars[length(candidate.vars)-1]])){
      candidate.vars <- candidate.vars[1:(length(candidate.vars)-1)]
    }
  }
  candidate.vars <- rev(candidate.vars) # revert order
  iter <- length(candidate.vars)
  
  
  ## Scoring stage
  pred.vec <- rep(NA, length=iter)
  pred.se.vec <- rep(NA, length=iter)
  names(pred.vec) <- names(pred.se.vec) <- paste('p=', candidate.vars, sep='')
  
  for(i in 1:iter){
    #print(i)
    seleted.vars <- vim.sorted[1:candidate.vars[i]]
    reduced_data <- data[,c(Y, names(seleted.vars))]
    
    # 5-fold CV
    kk <- 5
    cv.idx <- get.kfold.splits(Y=reduced_data$Y, k=kk, seed=123)
    cv.pred.vec.temp <- rep(NA, kk)
    
    for(j in 1:kk){ # 5-fold CV
      dat.train <- reduced_data[c(cv.idx[[j]]$training$case, cv.idx[[j]]$training$control),]
      dat.test <- reduced_data[c(cv.idx[[j]]$test$case, cv.idx[[j]]$test$control),]
      set.seed(123)
      fit.rf <- randomForest(factor(Y)~., data=dat.train, ntree=ntree)
      
      if(method=='pAC'){
        pred.Y <- predict(fit.rf, newdata=dat.test, type="response")
        conf.mat <- table(pred.Y, factor(dat.test$Y, levels=c(0,1)))
        cv.pred.vec.temp[j] <- (conf.mat[1,1]+conf.mat[2,2])/sum(conf.mat) # cv-ac (equivant to 1-cv.er)
        
      } else if(method=='pAUC' | method=='pAUC+US' | method=='pAUC+OS' | method=='pAUC+ST'){
        pred.Y <- predict(fit.rf, newdata=dat.test, type="prob")[, '1'] # prob for predicting case (0:control, 1:case)
        cv.pred.vec.temp[j] <- fast.auc(pred.Y, dat.test$Y, reverse.sign.if.nece = FALSE) # cv-auc
        
      }
    }
    
    pred.vec[i] <- mean(cv.pred.vec.temp[!is.infinite(cv.pred.vec.temp)])
  }
  
  # Select an optimal set of features based on CV-AUC
  best.idx <- which.max(pred.vec)
  best.pred <- pred.vec[best.idx]
  final.num.vars <- candidate.vars[best.idx]
  selected.vars <- names(vim.sorted[1:final.num.vars])
  selected.vars <- sort(selected.vars) # new
  
  return(list('final.set'=selected.vars, 'candidate.vars'=candidate.vars, 'pred'=pred.vec, 'vim.sorted'=vim.sorted, 'vim.se.sorted'=vim.se.sorted))
}

# Reference mehtods (Lasso and ElasticNet)
ref.method1 <- function(data, Y, family='binomial', method=c('LS','EN')) {
  
  # setting
  set.seed(123)
  Y <- data$Y
  dat.X <- data[, startsWith(colnames(data),'X')]
  for(ii in 1:ncol(dat.X)){ # account for categorical variables; dummy variables not working so that we need to calculate importance for each variable
    if(is.factor(dat.X[,ii])){dat.X[,ii] <- as.numeric(dat.X[,ii])} # forcing categorial to numeric
  }
  
  if(method=='LS'){
    # Lasso
    res.ls <- cv.glmnet(x=as.matrix(dat.X), y=as.matrix(Y), family=family, type.measure='auc', nfolds=5, alpha=1) # Lasso penalty
    vars.ls <- (coef(res.ls, s=res.ls$lambda.min) != 0)[-1]
    vars <- vars.ls
    
  } else if(method=='EN'){
    # Elastic
    train.idx <- sample(1:nrow(dat.X), ceiling(nrow(dat.X)*0.7)) # train:test=7:3
    test.idx <- setdiff(1:nrow(dat.X), train.idx)
    X.train <- dat.X[train.idx, ]; Y.train <- Y[train.idx]
    X.test <- dat.X[test.idx, ]; Y.test <- Y[test.idx]
    
    # tuning alpha
    alpha.tune <- list()
    for (i in 0:10) {
      alpha.temp <- paste0("alpha", i/10)
      alpha.tune[[alpha.temp]] <- cv.glmnet(x=as.matrix(X.train), y=as.matrix(Y.train), type.measure="auc", alpha=i/10, family=family, nfolds=5)
    }
    
    tune.res <- data.frame()
    for (i in 0:10) {
      alpha.temp <- paste0("alpha", i/10)
      predicted <- predict(alpha.tune[[alpha.temp]], s=alpha.tune[[alpha.temp]]$lambda.min, newx=as.matrix(X.test), type='response')
      auc.temp <- fast.auc(as.vector(predicted), Y.test, reverse.sign.if.nece = FALSE) # calculate AUC; no weights
      temp <- data.frame(alpha=i/10, auc.temp=auc.temp, alpha.temp=alpha.temp)
      tune.res <- rbind(tune.res, temp)
    }
    alpha.opt <- tune.res$alpha[which.max(tune.res$auc.temp)]
    
    # fit elastic net with optimal alpha
    res.el <- cv.glmnet(x=as.matrix(dat.X), y=as.matrix(Y), family=family, type.measure='auc', nfolds=5, alpha=alpha.opt) # elastic penalty
    vars.el <- (coef(res.el, s=res.el$lambda.min) != 0)[-1]
    vars <- vars.el
    
  }
  
  names(vars) <- colnames(dat.X)
  selected.vars <- colnames(dat.X)[vars]
  selected.vars <- sort(selected.vars) # new
  return(list('final.set'=selected.vars))
}

# Reference methods (Relief and SPEC)
ref.method2 <- function(data, Y, ntree, method=c('Relief','SPEC')){
  
  # Measure feature score
  if(method=='Relief'){
    # Relief
    set.seed(123)
    data$Y <- factor(data$Y)
    res.temp <- relief(Y~., data=data) # neighboursCount = 5, sampleSize = 10 by default
    sorted.vims <- res.temp$importance[order(res.temp$importance, decreasing=TRUE)]
    sorted.vars <- res.temp$attributes[order(res.temp$importance, decreasing=TRUE)]
    
  } else if(method=='SPEC'){
    # SPEC
    set.seed(123)
    data$Y <- factor(data$Y)
    dat.X <- data[, startsWith(colnames(data), 'X')]
    for(ii in 1:ncol(dat.X)){ # account for categorical variables
      if(is.factor(dat.X[,ii])){dat.X[,ii] <- as.numeric(dat.X[,ii])} # forcing categorial to numeric
    }
    dat.X <- as.matrix(dat.X)
    
    res.temp <- do.specs(X=dat.X, label=data$Y, ranking="method2") # method2 is robust by Zhao and Liu (2007)
    res.temp$attributes <- colnames(dat.X)
    sorted.vims <- res.temp$sscore[order(res.temp$sscore, decreasing=TRUE)]
    sorted.vars <- res.temp$attributes[order(res.temp$sscore, decreasing=TRUE)]
    
  }
  
  # Iterate fitting RF with different subsets of features=(1, top-10%, ..., top-90%, p)
  p <- ncol(data)-1
  var.set <- c(1,floor(seq(0.1,1,0.1)*p))
  iters <- length(var.set)
  pred.vec <- rep(NA, length=iters)
  names(pred.vec) <- paste0('p=',var.set)
  
  for(i in 1:iters){
    #print(i)
    seleted.vars <- sorted.vars[1:var.set[i]]
    reduced.data <- data[,c(Y, seleted.vars)]
    
    # 5-fold CV
    kk <- 5
    cv.idx <- get.kfold.splits(Y=reduced.data$Y, k=kk, seed=123)
    cv.pred.vec.temp <- rep(NA, kk)
    
    for(j in 1:kk){ # 5-fold CV
      dat.train <- reduced.data[c(cv.idx[[j]]$training$case, cv.idx[[j]]$training$control),]
      dat.test <- reduced.data[c(cv.idx[[j]]$test$case, cv.idx[[j]]$test$control),]
      set.seed(123)
      fit.rf <- randomForest(factor(Y)~., data=dat.train, ntree=ntree)
      pred.Y <- predict(fit.rf, newdata=dat.test, type="prob")[, '1'] # prob for predicting case (0:control, 1:case)
      cv.pred.vec.temp[j] <- fast.auc(pred.Y, dat.test$Y, reverse.sign.if.nece = FALSE) # cv-auc
    }
    
    pred.vec[i] <-  mean(cv.pred.vec.temp[!is.infinite(cv.pred.vec.temp)])
  }
  
  # Select an optimal set of features 
  best.idx <- which.max(pred.vec)
  best.pred <- pred.vec[best.idx]
  selected.vars <- sorted.vars[1:var.set[best.idx]]
  selected.vars <- sort(selected.vars) # new
  
  return(list('final.set'=selected.vars, 'candidate.vars'=sorted.vars, 'pred'=pred.vec, 'vim'=sorted.vims))
}
