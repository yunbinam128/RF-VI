# 1. Investigating simulation parameters
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



# 2. Measuring variable importance
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
  write.table(res.mat, file=paste0(res.path, filename), sep=',', row.names=TRUE, col.names=TRUE) # set res.path
}
