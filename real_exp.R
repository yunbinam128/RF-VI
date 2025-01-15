#rm(list=ls())
#code.path <- '/Users/sxh3040/Desktop/Paper/SHan/1.VIM/Code/' # change to your path where codes are saved
#source(paste(code.path, "data_import.R", sep=''))
res.path <- '/Users/sxh3040/Desktop/Paper/SHan/1.VIM/Result/'

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

for(i in 1:length(data.list)){ # 1:length(data.list) 
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
for(i in 1:length(dat.name)){ # 1:length(total.name)
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
  write.table(auc.mat, file=paste0(res.path, dat.name[i], '_cvauc.txt'), col.names=TRUE, row.names=FALSE, sep=',')
}



# 3. Box-plot in Section 4.1
auc.mat <- matrix(NA, nrow=length(dat.name)*100, ncol=length(prop.methods))
colnames(auc.mat) <- prop.methods
for(i in 1:length(dat.name)){
  print(i)
  res <- read.table(paste0(res.path, dat.name[i], '_cvauc.txt'), sep=',', header=TRUE)
  res <- res[,c('pAC.CVER','pAUC','pAUC.US','pAUC.OS','pAUC.ST')]
  auc.mat[(1+100*(i-1)):(100*i),] <- as.matrix(res)
}

par(mfrow=c(1,2)) # 6 by 10
# balanced (IR<2)
boxplot(auc.mat, ylab='CV-AUC', xaxt='n', main='IR<2')
axis(1, at=1:ncol(auc.mat), labels=FALSE)
text(x=1:ncol(auc.mat), y=par("usr")[3]-0.04, labels=c('pAC','pAUC','pAUC+US','pAUC+OS','pAUC+ST'), xpd=TRUE, cex=0.8)
points(apply(auc.mat, 2, mean, na.rm=T), type='o', cex=0.8, pch=20)
#text(1:ncol(auc.mat), y=apply(auc.mat, 2, mean, na.rm=T)-0.015, labels=round(apply(auc.mat, 2, mean, na.rm=T),4), cex=0.8)
text(1:ncol(auc.mat), y=apply(auc.mat, 2, mean, na.rm=T)-0.015, labels=c(0.9255,0.9263,0.9267,0.9252,0.9260), cex=0.8)

# imbalanced (IR>2)
boxplot(auc.mat, ylab='CV-AUC', xaxt='n', main='IR>2')
axis(1, at=1:ncol(auc.mat), labels=FALSE)
text(x=1:ncol(auc.mat), y=par("usr")[3]-0.04, labels=c('pAC','pAUC','pAUC+US','pAUC+OS','pAUC+ST'), xpd=TRUE, cex=0.8)
points(apply(auc.mat, 2, mean, na.rm=T), type='o', cex=0.8, pch=20)
text(1:ncol(auc.mat), y=apply(auc.mat, 2, mean, na.rm=T)-0.015, labels=round(apply(auc.mat, 2, mean, na.rm=T),4), cex=0.8)



# 4. Box-plot in Section 4.2
methods <- c('SPEC','Relief','LS','EN','pAUC','pAUC+US','pAUC+OS','pAUC+ST')
auc.mat <- matrix(NA, nrow=length(dat.name)*100, ncol=length(methods))
colnames(auc.mat) <- methods
for(i in 1:length(dat.name)){
  print(i)
  res.ref <- read.table(paste0(res.path, dat.name[i], '_cvauc_ref.txt'), sep=',', header=TRUE)
  res.prop <- read.table(paste0(res.path, dat.name[i], '_cvauc.txt'), sep=',', header=TRUE)
  res <- cbind(res.ref[,c('SPEC','Relief','LS','EN')], res.prop[,c('pAUC','pAUC.US','pAUC.OS','pAUC.ST')])
  auc.mat[(1+100*(i-1)):(100*i),] <- as.matrix(res)
}

par(mfrow=c(1,1)) # 6 by 8
boxplot(auc.mat, ylab='CV-AUC', xaxt='n')
axis(1, at=1:ncol(auc.mat), labels=FALSE)
text(x=1:ncol(auc.mat), y=par("usr")[3]-0.04, labels=methods, xpd=TRUE, cex=0.8)
points(apply(auc.mat, 2, mean, na.rm=T), type='o', cex=0.8, pch=20)
text(1:ncol(auc.mat), y=apply(auc.mat, 2, mean, na.rm=T)-0.015, labels=round(apply(auc.mat, 2, mean, na.rm=T),4), cex=0.8)




# 4. CV-AUC table
methods <- c('SPEC','Relief','LS','EN','pAUC','pAUC+US','pAUC+OS','pAUC+ST')
mean.auc.mat <- matrix(NA, nrow=length(dat.name), ncol=length(methods))
sd.auc.mat <- matrix(NA, nrow=length(dat.name), ncol=length(methods))
rownames(mean.auc.mat) <- rownames(sd.auc.mat) <- dat.name
colnames(mean.auc.mat) <- colnames(sd.auc.mat) <- methods
for(i in 1:length(dat.name)){
  print(i)
  res.ref <- read.table(paste0(res.path, dat.name[i], '_cvauc_ref.txt'), sep=',', header=TRUE)
  res.prop <- read.table(paste0(res.path, dat.name[i], '_cvauc.txt'), sep=',', header=TRUE)
  res <- cbind(res.ref[,c('SPEC','Relief','LS','EN')], res.prop[,c('pAUC','pAUC.US','pAUC.OS','pAUC.ST')])
  mean.auc.mat[i,] <- round(apply(res, 2, mean), 4)
  sd.auc.mat[i,] <- round(apply(res, 2, sd), 4)
}
mean.auc.mat
sd.auc.mat





# 4. paired t-test and two-sample t-test
methods <- c(ref.methods,'pAUC','pAUC+US','pAUC+OS','pAUC+ST')
comps <- list(c('SPEC','Relief'),c('SPEC','LS'),c('SPEC','EN'),c('SPEC','pAUC'),c('SPEC','pAUC+US'),c('SPEC','pAUC+OS'),c('SPEC','pAUC+ST'),
              c('Relief','LS'),c('Relief','EN'),c('Relief','pAUC'),c('Relief','pAUC+US'),c('Relief','pAUC+OS'),c('Relief','pAUC+ST'),
              c('LS','EN'),c('LS','pAUC'), c('LS','pAUC+US'), c('LS','pAUC+OS'), c('LS','pAUC+ST'),
              c('EN','pAUC'),c('EN','pAUC+US'),c('EN','pAUC+OS'),c('EN','pAUC+ST'),
              c('pAUC','pAUC+US'), c('pAUC','pAUC+OS'), c('pAUC','pAUC+ST'),
              c('pAUC+US','pAUC+OS'), c('pAUC+US','pAUC+ST'),
              c('pAUC+OS','pAUC+ST'))
comps.vec <- c('SPEC vs Relief','SPEC vs LS','SPEC vs EN','SPEC vs pAUC','SPEC vs pAUC+US','SPEC vs pAUC+OS','SPEC vs pAUC+ST',
               'Relief vs LS','Relief vs EN','Relief vs pAUC','Relief vs pAUC+US','Relief vs pAUC+OS','Relief vs pAUC+ST',
               'LS vs EN','LS vs pAUC', 'LS vs pAUC+US', 'LS vs pAUC+OS', 'LS vs pAUC+ST',
               'EN vs pAUC','EN vs pAUC+US','EN vs pAUC+OS','EN vs pAUC+ST',
               'pAUC vs pAUC+US', 'pAUC vs pAUC+OS', 'pAUC vs pAUC+ST',
               'pAUC+US vs pAUC+OS', 'pAUC+US vs pAUC+ST',
               'pAUC+OS vs pAUC+ST')

test.mat <- as.data.frame(matrix(NA, nrow=length(dat.name)*length(comps.vec), ncol=4))
test.mat[,1] <- rep(dat.name, times=length(comps.vec))
test.mat[,2] <- rep(comps.vec, each=length(dat.name))
colnames(test.mat) <- c('id','test','t-stat','p-value')
for(i in 1:length(dat.name)){
  print(i)
  res.ref <- read.table(paste0(res.path, dat.name[i], '_cvauc_ref.txt'), sep=',', header=TRUE)
  res.prop <- read.table(paste0(res.path, dat.name[i], '_cvauc.txt'), sep=',', header=TRUE)
  res <- cbind(res.ref[,c('SPEC','Relief','LS','EN')], res.prop[,c('pAUC','pAUC.US','pAUC.OS','pAUC.ST')])
  colnames(res) <- methods
  
  for(j in 1:length(comps)){
    print(comps[[j]])
    # t-test
    temp <- t.test(res[,comps[[j]][1]], res[,comps[[j]][2]], paired=TRUE, alternative='two.sided') # paired t-test
    #temp <- wilcox.test(res[,comps[[j]][1]], res[,comps[[j]][2]], paired=TRUE, alternative='two.sided') # wilcoxon signed-rank test
    test.mat[length(dat.name)*(j-1)+i,3:4] <- c(round(temp$statistic, 4), round(temp$p.value, 4))
    
    # Bayesian test
    #if(sum(res[,comps[[j]][1]] != res[,comps[[j]][2]]) != 0){
    #  temp <- ttestBF(x=res[,comps[[j]][1]], y=res[,comps[[j]][2]], paired=TRUE) # Bayesian t-test
    #  test.mat[length(dat.name)*(j-1)+i,3] <- unname(as.vector(temp))
    #}
  }
}

# Paired t-test
win.loss.table <- matrix(NA, nrow=length(methods), ncol=length(methods))
rownames(win.loss.table) <- colnames(win.loss.table) <- methods
sig.table <- matrix(NA, nrow=length(methods), ncol=length(methods))
rownames(sig.table) <- colnames(sig.table) <- methods
row.idx <- c(2:8, 3:8, 4:8, 5:8, 6:8, 7:8, 8)
col.idx <- c(rep(1,7), rep(2,6), rep(3,5), rep(4,4), rep(5,3), rep(6,2), rep(7,1))
for(i in 1:length(comps.vec)){
  print(i)
  temp.mat <- test.mat[test.mat[,'test']==comps.vec[i], ]
  col.win <- sum(temp.mat$`t-stat` > 0, na.rm=TRUE)
  row.win <- sum(temp.mat$`t-stat` < 0, na.rm=TRUE)
  col.sig.win <- sum((temp.mat$`t-stat` > 0) & (temp.mat$`p-value` < 0.05), na.rm=TRUE)
  row.sig.win <- sum((temp.mat$`t-stat` < 0) & (temp.mat$`p-value` < 0.05), na.rm=TRUE)
  win.loss.table[row.idx[i], col.idx[i]] <- paste0(col.win, ' (', col.sig.win, ')')
  win.loss.table[col.idx[i], row.idx[i]] <- paste0(row.win, ' (', row.sig.win, ')')
  sig.table[row.idx[i], col.idx[i]] <- col.sig.win
  sig.table[col.idx[i], row.idx[i]] <- row.sig.win
}
win.loss.table
sig.table

# Win-loss table
win.loss <- matrix(NA, nrow=length(methods), ncol=3)
rownames(win.loss) <- methods; colnames(win.loss) <- c('Win', 'Loss', 'Win-Loss')
for(i in 1:length(methods)){
  print(i)
  win.loss[i,1] <- sum(sig.table[,i], na.rm=TRUE)
  win.loss[i,2] <- sum(sig.table[i,], na.rm=TRUE)
  win.loss[i,3] <- win.loss[i,1] - win.loss[i,2]
}
win.loss






#####
library(devtools)
install_github('ramhiser/datamicroarray')
library(chiaretti)
library(datamicroarray)

data(su)
dim(su$x)
table(su$y)
y <- as.vector(su$y)
#idx <- y=='placenta' | y=='blood'
#dat.x <- christensen$x[idx,]
#y <- y[idx]
y[y!='n'] <- 0
y[y=='n'] <- 1
y <- as.numeric(y)
table(y)
write.table(cbind(alon$x,y), '/Users/sxh3040/Downloads/col.txt', col.names=TRUE, row.names=TRUE, sep=',')


temp <- read.table('/Users/sxh3040/Downloads/pre.csv', header=T, sep='')
dim(temp)
table(temp$Label)


temp <- read.table('/Users/sxh3040/Desktop/Paper/SHan/4.DRF_HD/Data/oestrogen_activity.txt', header=T, sep=' ')
table(temp$class)
temp <- temp[,2:452]
write.table(temp,'/Users/sxh3040/Downloads/dar.txt', col.names=T, sep=',')








# Representative examples
methods <- c('pAC','pAUC','pAUC+US','pAUC+OS','pAUC+ST')
mean.auc.mat <- matrix(NA, nrow=length(dat.name), ncol=length(methods))
sd.auc.mat <- matrix(NA, nrow=length(dat.name), ncol=length(methods))
rownames(mean.auc.mat) <- rownames(sd.auc.mat) <- dat.name
colnames(mean.auc.mat) <- colnames(sd.auc.mat) <- methods
for(i in 1:length(dat.name)){
  print(i)
  res.prop <- read.table(paste0(res.path, dat.name[i], '_cvauc.txt'), sep=',', header=TRUE)
  res <- res.prop[,c('pAC.CVER','pAUC','pAUC.US','pAUC.OS','pAUC.ST')]
  mean.auc.mat[i,] <- round(apply(res, 2, mean), 4)
  sd.auc.mat[i,] <- round(apply(res, 2, sd), 4)
}
mean.auc.mat
sd.auc.mat


# Import plotrix library 
library("plotrix") 

# hil
i <- 20 # 4, 25
res.vi <- readRDS(paste(res.path, dat.name[i], '_varsec.RData', sep=''))
dat <- data.list[[i]]
p <- ncol(dat)-1
methods <- c('pAC','pAUC','pAUC+ST')  # change methods
for(k in 1:length(methods)){
  print(k)
  method <- methods[k]
  if(method=='pAC'){res <- res.vi$`pAC-CVER`}
  if(method=='pAUC'){res <- res.vi$pAUC}
  if(method=='pAUC+ST'){res <- res.vi$`pAUC+ST`}
  
  vi <- res$vim.sorted
  vi.se <- res$vim.se.sorted
  vi.lb <- vi - vi.se # u=1
  vi.ub <- vi + vi.se # u=1
  #idx <- res$candidate.vars
  #pred <- res$pred
  idx <- res$candidate.vars[2:length(res$candidate.vars)]
  pred <- round(res$pred[2:length(res$candidate.vars)],3)
  
  if(method=='pAC'){
    par(mfrow=c(3,1))
    par(mar=c(2, 4, 2, 2))
    plotCI(x=1:p, y=vi, li=vi.lb, ui=vi.ub, 
           pch=20, cex=0.5, ylab='VI', xaxt='n', ylim=c(min(vi.lb),max(vi.ub)+0.002), xlab='', main=method)
    axis(1, at=1:length(vi), labels=FALSE)
    text(x=c(1:p)[(1:p%%2)==1], y=par("usr")[3]-0.0005, labels=names(vi)[(1:p%%2)==1], xpd=TRUE, cex=0.8)
    text(x=c(1:p)[(1:p%%2)==0], y=par("usr")[3]-0.0008, labels=names(vi)[(1:p%%2)==0], xpd=TRUE, cex=0.8)
    #text(x=idx, y=par("usr")[3]-0.0005, labels=names(vi)[idx], xpd=TRUE, cex=1)
    abline(h=vi.ub[idx], lty='dotted', lwd=0.5)
    abline(v=idx+0.5, lty='dotted', col='blue')
    text(idx, y=max(vi.ub)+0.002, labels=paste0('p=',idx), cex=1)
    text(idx, y=max(vi.ub)+0.0016, labels='CV-AC=', cex=1)
    text(idx, y=max(vi.ub)+0.0012, labels=pred, cex=1)
    
  } else{
    plotCI(x=1:p, y=vi, li=vi.lb, ui=vi.ub, 
           pch=20, cex=0.5, ylab='VI', xaxt='n', ylim=c(min(vi.lb),max(vi.ub)+0.002), xlab='', main=method)
    axis(1, at=1:length(vi), labels=FALSE)
    text(x=c(1:p)[(1:p%%2)==1], y=par("usr")[3]-0.0005, labels=names(vi)[(1:p%%2)==1], xpd=TRUE, cex=0.8)
    text(x=c(1:p)[(1:p%%2)==0], y=par("usr")[3]-0.0008, labels=names(vi)[(1:p%%2)==0], xpd=TRUE, cex=0.8)
    #text(x=idx, y=par("usr")[3]-0.0005, labels=names(vi)[idx], xpd=TRUE, cex=1)
    abline(h=vi.ub[idx], lty='dotted', lwd=0.5)
    abline(v=idx+0.5, lty='dotted', col='blue')
    text(idx, y=max(vi.ub)+0.002, labels=paste0('p=',idx), cex=1)
    text(idx, y=max(vi.ub)+0.0016, labels='CV-AUC=', cex=1)
    text(idx, y=max(vi.ub)+0.0012, labels=pred, cex=1)
  }
}



# yea4
i <- 21 # 4, 25
res.vi <- readRDS(paste(res.path, dat.name[i], '_varsec.RData', sep=''))
dat <- data.list[[i]]
p <- ncol(dat)-1
methods <- c('pAC','pAUC','pAUC+ST')  # change methods
for(k in 1:length(methods)){
  print(k)
  method <- methods[k]
  if(method=='pAC'){res <- res.vi$`pAC-CVER`}
  if(method=='pAUC'){res <- res.vi$pAUC}
  if(method=='pAUC+ST'){res <- res.vi$`pAUC+ST`}
  
  vi <- res$vim.sorted
  vi.se <- res$vim.se.sorted
  vi.lb <- vi - vi.se # u=1
  vi.ub <- vi + vi.se # u=1
  idx <- res$candidate.vars
  pred <- round(res$pred,3)
  
  if(method=='pAC'){
    #idx <- res$candidate.vars[2:length(res$candidate.vars)]
    #pred <- round(res$pred[2:length(res$candidate.vars)],3)
    
    par(mfrow=c(3,1))
    par(mar=c(2, 4, 2, 2))
    plotCI(x=1:p, y=vi, li=vi.lb, ui=vi.ub, 
           pch=20, cex=0.5, ylab='VI', xaxt='n', ylim=c(min(vi.lb),max(vi.ub)+0.005), xlab='', main=method)
    axis(1, at=1:length(vi), labels=FALSE)
    text(x=1:p, y=par("usr")[3]-0.0015, labels=names(vi), xpd=TRUE, cex=1)
    #text(x=idx, y=par("usr")[3]-0.0005, labels=names(vi)[idx], xpd=TRUE, cex=1)
    abline(h=vi.ub[idx], lty='dotted', lwd=0.5)
    abline(v=idx+0.5, lty='dotted', col='blue')
    text(idx, y=max(vi.ub)+0.005, labels=paste0('p=',idx), cex=1)
    text(idx, y=max(vi.ub)+0.004, labels='CV-AC=', cex=1)
    text(idx, y=max(vi.ub)+0.003, labels=pred, cex=1)
    
  } else if(method=='pAUC'){
    
    plotCI(x=1:p, y=vi, li=vi.lb, ui=vi.ub, 
           pch=20, cex=0.5, ylab='VI', xaxt='n', ylim=c(min(vi.lb),max(vi.ub)+0.035), xlab='', main=method)
    axis(1, at=1:length(vi), labels=FALSE)
    text(x=1:p, y=par("usr")[3]-0.01, labels=names(vi), xpd=TRUE, cex=1)
    #text(x=idx, y=par("usr")[3]-0.0005, labels=names(vi)[idx], xpd=TRUE, cex=1)
    abline(h=vi.ub[idx], lty='dotted', lwd=0.5)
    abline(v=idx+0.5, lty='dotted', col='blue')
    text(idx, y=max(vi.ub)+0.035, labels=paste0('p=',idx), cex=1)
    text(idx, y=max(vi.ub)+0.028, labels='CV-AUC=', cex=1)
    text(idx, y=max(vi.ub)+0.021, labels=pred, cex=1)
    
  } else if(method=='pAUC+ST'){
    
    plotCI(x=1:p, y=vi, li=vi.lb, ui=vi.ub, 
           pch=20, cex=0.5, ylab='VI', xaxt='n', ylim=c(min(vi.lb),max(vi.ub)+0.08), xlab='', main=method)
    axis(1, at=1:length(vi), labels=FALSE)
    text(x=1:p, y=par("usr")[3]-0.02, labels=names(vi), xpd=TRUE, cex=1)
    #text(x=idx, y=par("usr")[3]-0.0005, labels=names(vi)[idx], xpd=TRUE, cex=1)
    abline(h=vi.ub[idx], lty='dotted', lwd=0.5)
    abline(v=idx+0.5, lty='dotted', col='blue')
    text(idx, y=max(vi.ub)+0.08, labels=paste0('p=',idx), cex=1)
    text(idx, y=max(vi.ub)+0.06, labels='CV-AUC=', cex=1)
    text(idx, y=max(vi.ub)+0.04, labels=pred, cex=1)
    
  }
}





# Bayesian test
library(BayesFactor)

methods <- c(ref.methods,'pAUC','pAUC+US','pAUC+OS','pAUC+ST')
comps <- list(c('SPEC','Relief'),c('SPEC','LS'),c('SPEC','EN'),c('SPEC','pAUC'),c('SPEC','pAUC+US'),c('SPEC','pAUC+OS'),c('SPEC','pAUC+ST'),
              c('Relief','LS'),c('Relief','EN'),c('Relief','pAUC'),c('Relief','pAUC+US'),c('Relief','pAUC+OS'),c('Relief','pAUC+ST'),
              c('LS','EN'),c('LS','pAUC'), c('LS','pAUC+US'), c('LS','pAUC+OS'), c('LS','pAUC+ST'),
              c('EN','pAUC'),c('EN','pAUC+US'),c('EN','pAUC+OS'),c('EN','pAUC+ST'),
              c('pAUC','pAUC+US'), c('pAUC','pAUC+OS'), c('pAUC','pAUC+ST'),
              c('pAUC+US','pAUC+OS'), c('pAUC+US','pAUC+ST'),
              c('pAUC+OS','pAUC+ST'))
comps.vec <- c('SPEC vs Relief','SPEC vs LS','SPEC vs EN','SPEC vs pAUC','SPEC vs pAUC+US','SPEC vs pAUC+OS','SPEC vs pAUC+ST',
               'Relief vs LS','Relief vs EN','Relief vs pAUC','Relief vs pAUC+US','Relief vs pAUC+OS','Relief vs pAUC+ST',
               'LS vs EN','LS vs pAUC', 'LS vs pAUC+US', 'LS vs pAUC+OS', 'LS vs pAUC+ST',
               'EN vs pAUC','EN vs pAUC+US','EN vs pAUC+OS','EN vs pAUC+ST',
               'pAUC vs pAUC+US', 'pAUC vs pAUC+OS', 'pAUC vs pAUC+ST',
               'pAUC+US vs pAUC+OS', 'pAUC+US vs pAUC+ST',
               'pAUC+OS vs pAUC+ST')

test.mat <- as.data.frame(matrix(NA, nrow=length(dat.name)*length(comps.vec), ncol=5))
test.mat[,1] <- rep(dat.name, times=length(comps.vec))
test.mat[,2] <- rep(comps.vec, each=length(dat.name))
colnames(test.mat) <- c('id','test','BF','CV-AUC_col','CV-AUC_row')
for(i in 1:length(dat.name)){
  print(i)
  res.ref <- read.table(paste0(res.path, dat.name[i], '_cvauc_ref.txt'), sep=',', header=TRUE)
  res.prop <- read.table(paste0(res.path, dat.name[i], '_cvauc.txt'), sep=',', header=TRUE)
  res <- cbind(res.ref[,c('SPEC','Relief','LS','EN')], res.prop[,c('pAUC','pAUC.US','pAUC.OS','pAUC.ST')])
  colnames(res) <- methods
  
  for(j in 1:length(comps)){
    print(comps[[j]])
    
    # Bayesian test
    if(sum(res[,comps[[j]][1]] != res[,comps[[j]][2]]) != 0){
      test.mat[length(dat.name)*(j-1)+i,'CV-AUC_col'] <- mean(res[,comps[[j]][1]])
      test.mat[length(dat.name)*(j-1)+i,'CV-AUC_row'] <- mean(res[,comps[[j]][2]])
      temp <- ttestBF(x=res[,comps[[j]][1]], y=res[,comps[[j]][2]], paired=TRUE) # Bayesian t-test
      test.mat[length(dat.name)*(j-1)+i,'BF'] <- unname(as.vector(temp))
      #print(unname(as.vector(temp)))
    }
  }
}

# Paired Bayesian test
win.loss.table <- matrix(NA, nrow=length(methods), ncol=length(methods))
rownames(win.loss.table) <- colnames(win.loss.table) <- methods
sig.table <- matrix(NA, nrow=length(methods), ncol=length(methods))
rownames(sig.table) <- colnames(sig.table) <- methods
row.idx <- c(2:8, 3:8, 4:8, 5:8, 6:8, 7:8, 8)
col.idx <- c(rep(1,7), rep(2,6), rep(3,5), rep(4,4), rep(5,3), rep(6,2), rep(7,1))
for(i in 1:length(comps.vec)){
  print(i)
  temp.mat <- test.mat[test.mat[,'test']==comps.vec[i], ]
  col.win <- sum(temp.mat$`CV-AUC_col` > temp.mat$`CV-AUC_row`, na.rm=TRUE)
  row.win <- sum(temp.mat$`CV-AUC_col` < temp.mat$`CV-AUC_row`, na.rm=TRUE)
  col.sig.win <- sum((temp.mat$BF > 10) & (temp.mat$`CV-AUC_col` > temp.mat$`CV-AUC_row`), na.rm=TRUE)
  row.sig.win <- sum((temp.mat$BF > 10) & (temp.mat$`CV-AUC_col` < temp.mat$`CV-AUC_row`), na.rm=TRUE)
  win.loss.table[row.idx[i], col.idx[i]] <- paste0(col.win, ' (', col.sig.win, ')')
  win.loss.table[col.idx[i], row.idx[i]] <- paste0(row.win, ' (', row.sig.win, ')')
  sig.table[row.idx[i], col.idx[i]] <- col.sig.win
  sig.table[col.idx[i], row.idx[i]] <- row.sig.win
}
win.loss.table
sig.table

# Win-loss table
win.loss <- matrix(NA, nrow=length(methods), ncol=3)
rownames(win.loss) <- methods; colnames(win.loss) <- c('Win', 'Loss', 'Win-Loss')
for(i in 1:length(methods)){
  print(i)
  win.loss[i,1] <- sum(sig.table[,i], na.rm=TRUE)
  win.loss[i,2] <- sum(sig.table[i,], na.rm=TRUE)
  win.loss[i,3] <- win.loss[i,1] - win.loss[i,2]
}
win.loss
