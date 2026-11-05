"""Monotone sigmoid calibration of cross-fitted binary probabilities."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


class MonotoneSigmoid:
    def __init__(self, regularization=.01):
        self.regularization = regularization

    def fit(self, probability, target):
        p, y = np.asarray(probability,float), np.asarray(target)
        if p.ndim!=1 or p.shape!=y.shape or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():
            raise ValueError("Matched finite OOF binary probabilities and labels required")
        self.n_, self.positive_ = len(y), int(y.sum())
        if len(np.unique(y))<2:
            self.slope_, self.intercept_, self.status_ = 1.,0.,"identity_single_class"
            return self
        x=logit(np.clip(p,1e-12,1-1e-12)); alpha=self.regularization
        if not np.isfinite(alpha) or alpha<0:
            raise ValueError("Nonnegative finite regularization required")
        def objective(parameters):
            w,c=parameters;z=w*x+c;error=expit(z)-y
            loss=np.mean(np.logaddexp(0,z)-y*z)+alpha*((w-1)**2+c*c)/2
            return loss,np.array([np.mean(error*x)+alpha*(w-1),np.mean(error)+alpha*c])
        result=minimize(objective,[1.,0.],jac=True,method="L-BFGS-B",bounds=[(.01,10),(-15,15)],options={"maxiter":500,"ftol":1e-12})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("Calibration optimizer failed: "+str(result.message))
        self.slope_, self.intercept_ = map(float,result.x)
        self.status_="fitted_monotone_sigmoid"
        return self

    def predict(self, probability):
        p=np.asarray(probability,float)
        if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError("Valid binary probabilities required")
        if self.status_=="identity_single_class":return p.copy()
        return expit(self.slope_*logit(np.clip(p,1e-12,1-1e-12))+self.intercept_)

    def metadata(self):
        return dict(slope=self.slope_,intercept=self.intercept_,status=self.status_,n=self.n_,positive=self.positive_,regularization=self.regularization)
