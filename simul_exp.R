rm(list=ls())
code.path <- '/Users/sxh3040/Desktop/Paper/SHan/1.VIM/Code/' # change to your path where codes are saved
source(paste(code.path, "helper_vim.R", sep=''))

res.path <- '/Users/sxh3040/Desktop/Paper/SHan/1.VIM/Result/'



# 0. Investigating simulation parameters
N <- c(50, 100, 250, 500)
IR <- c(1, 2, 10, 20)
methods <- c("pAC", "pAUC", "pAUC+US", "pAUC+OS", "pAUC+ST")
method <- methods[5]
iter <- 500
print(c('method'=method, 'iter'=iter))

simul.mat <- matrix(NA, nrow=length(N)*length(IR), ncol=4)
colnames(simul.mat) <- c('n', 'ir', 'n1', 'n0')
for(i in 1:length(N)){
  n <- N[i]
  for(j in 1:length(IR)){
    ir <- IR[j]
    n1 <- floor(n*(1/(ir+1))) ; n0 <- n-n1
    simul.mat[4*(i-1)+j,] <- c(n, ir, n1, n0)
  }
}
simul.mat



# 1. Measuring variable importance
for(i in 1:nrow(simul.mat)){ 
  print(i)
  n <- simul.mat[i,'n']
  ir <- simul.mat[i,'ir']
  n1 <- simul.mat[i,'n1']
  n0 <- simul.mat[i,'n0']
  
  # Variable importance measure
  res.mat <- matrix(NA, nrow=iter, ncol=30) # 30 predictors in simulated data
  row.names(res.mat) <- 1:iter
  colnames(res.mat) <- paste('X', 1:30, sep='')
  for(j in 1:iter){ 
    print(j)
    # Generate simulation data
    seed <- j
    dat <- sim.imbalanced(n1=n1, n0=n0, seed=seed)
    p <- ncol(dat)-1
    
    # Different methods for measuring variable importance
    if(method=='pAC'){
      # OOB-AC
      res.mat[j,] <- get.vim(data=dat, Y='Y', method='pAC')$oob.pred
      
    } else if(method=='pAUC'){
      # OOB-AUC
      res.mat[j,] <- get.vim(data=dat, Y='Y', method='pAUC')$oob.pred
      
    } else if(method=='pAUC+US'){
      # OOB-AUC with under-sampling
      res.mat[j,] <- get.vim(data=dat, Y='Y', method='pAUC+US')$oob.pred
      
    } else if(method=='pAUC+OS'){
      # OOB-AUC with over-sampling
      res.mat[j,] <- get.vim(data=dat, Y='Y', method='pAUC+OS')$oob.pred
      
    } else if(method=='pAUC+ST'){
      # OOB-AUC with SMOTE-sampling 
      res.mat[j,] <- get.vim(data=dat, Y='Y', method='pAUC+ST')$oob.pred
    }
  }
  
  filename <- paste0('res_n', n, '_ir', ir, '_', method, '.txt')
  write.table(res.mat, file=paste0(res.path, filename), sep=',', row.names=TRUE, col.names=TRUE)
}



##################################################################################################



# Summarize results
# 1. AUC
N <- c(50, 100, 250, 500)
IR <- c(1, 2, 10, 20)
methods <- c("pAC", "pAUC", "pAUC+US", "pAUC+OS", "pAUC+ST")
iter <- 500

auc.mat <- matrix(NA, nrow=length(N)*length(IR)*length(methods), ncol=1)
setting <- paste('N=', rep(N, each=length(IR)), ', IR=', IR, sep='')
rownames(auc.mat) <- paste(rep(setting, each=length(methods)), ', ', methods, sep='')
true.Y <- c(rep(1,15), rep(0,15))
for(i in 1:length(N)){
  print(i)
  n <- N[i]
  for(j in 1:length(IR)){
    print(j)
    ir <- IR[j]
    for(k in 1:length(methods)){
      filename <- paste('res_n', n, '_ir', ir, '_', methods[k], '.txt', sep='')
      res.temp <- read.table(paste0(res.path, filename), header=TRUE, sep=',', nrows=500)
      
      auc.vec <- c()
      for(l in 1:nrow(res.temp)){
        pred.Y <- res.temp[l,]
        auc.vec[l] <- fast.auc(pred.Y, true.Y, reverse.sign.if.nece = FALSE)
      }
      
      auc.mat[20*(i-1)+5*(j-1)+k,] <- mean(auc.vec)
    }
  }
}
auc.mat



# 2. Box-plot of AUC
methods <- c("pAC", "pAUC", "pAUC+US", "pAUC+OS", "pAUC+ST")
N <- c(50, 100, 250, 500)
IR <- c(1, 2, 10, 20)
ir <- IR[4]

iter <- 500
true.Y <- c(rep(1,15), rep(0,15))
for(i in 1:length(N)){
  print(i)
  n <- N[i]
  auc.mat <- matrix(NA, nrow=iter, ncol=length(methods))
  colnames(auc.mat) <- methods
  
  for(k in 1:length(methods)){
    #print(k)
    filename <- paste('res_n', n, '_ir', ir, '_', methods[k], '.txt', sep='')
    res.temp <- read.table(paste0(res.path, filename), header=TRUE, sep=',')
    auc.vec <- c()
    for(l in 1:nrow(res.temp)){
      pred.Y <- res.temp[l,]
      auc.vec[l] <- fast.auc(pred.Y, true.Y, reverse.sign.if.nece = FALSE)
    }
    auc.mat[,k] <- auc.vec
  }
  
  # plotting
  #par(mfrow=c(4,4))
  #par(oma=c(1,1,1,1)) # all sides have 3 lines of space
  #par(mar=c(3,2,2,1) + 0.1)
  if(i==1){
    boxplot(auc.mat, main=paste0('N=',n, ' & IR=',ir), ylim=c(0,1), xaxt='n'); abline(h=0.5, lty='dotted')
    mtext("AUC", side=2, line=2, cex=0.70)
  } else{
    boxplot(auc.mat, main=paste0('N=',n, ' & IR=',ir), ylim=c(0,1), xaxt='n'); abline(h=0.5, lty='dotted')
  }
  axis(1, at=1:ncol(auc.mat), labels=FALSE)
  text(x=1:ncol(auc.mat), y=par("usr")[3]-0.07, labels=methods, xpd=TRUE, srt=20, adj=c(0.85,1), cex=0.8)
  
}




# 3. Box-plot of VI
methods <- c("pAC", "pAUC", "pAUC+US", "pAUC+OS", "pAUC+ST")
N <- c(50, 100, 250, 500)
n <- N[4]
IR <- c(1, 2, 10, 20)
ir <- IR[4]
print(c('n'=n, 'ir'=ir))

for(k in 1:length(methods)){
  print(k)
  filename <- paste('res_n', n, '_ir', ir, '_', methods[k], '.txt', sep='')
  res.temp <- read.table(paste0(res.path, filename), header=TRUE, sep=',')
  
  strong <- unname(unlist(c(res.temp[,1:5])))
  moderate <- unname(unlist(c(res.temp[,6:10])))
  weak <- unname(unlist(c(res.temp[,11:15])))
  noise <- unname(unlist(c(res.temp[,16:30])))
  
  # plotting
  if(k==1){par(mfrow=c(3,2))}
  boxplot(cbind(Strong=strong, Moderate=moderate, Weak=weak, Noise=noise), main=methods[k])
  mtext("VI", side=2, line=2, cex=0.7)
}



###



# 4. Table for VI
methods <- c("pAC", "pAUC", "pAUC+US", "pAUC+OS", "pAUC+ST")
N <- c(50, 100, 250, 500)
n <- N[1]
IR <- c(1, 2, 10, 20)
ir <- IR[1]
print(c('n'=n, 'ir'=ir))

vi.mat <- matrix(NA, nrow=length(methods), ncol=7)
rownames(vi.mat) <- methods; colnames(vi.mat) <- c('strong','moderate','weak','noise','strong/noise','moderate/noise','weak/noise')

for(k in 1:length(methods)){
  print(k)
  filename <- paste('res_n', n, '_ir', ir, '_', methods[k], '.txt', sep='')
  res.temp <- read.table(paste0(res.path, filename), header=TRUE, sep=',')
  
  strong <- unname(unlist(c(res.temp[,1:5])))
  moderate <- unname(unlist(c(res.temp[,6:10])))
  weak <- unname(unlist(c(res.temp[,11:15])))
  noise <- unname(unlist(c(res.temp[,16:30])))
  
  #noise.0 <- noise
  #noise.0[noise.0 <0] <- 0
  
  ratio <- c(mean(strong/noise),mean(moderate/noise),mean(weak/noise))
  vi.mat[k,] <- c(mean(strong), mean(moderate), mean(weak), mean(noise), ratio)
  
}
vi.mat
