import numpy as np
import scipy.linalg as la

check_finite = True

def cho_inv(A):
    L,low = la.cho_factor(A, check_finite=check_finite)
    B = np.eye(L.shape[0])
    B = la.cho_solve((L,low),B,overwrite_b=True,check_finite=check_finite)
    return B

def cho_solve(A,B):
    L,low = la.cho_factor(A, check_finite=check_finite)
    C = la.cho_solve((L,low),B,overwrite_b=False, check_finite=check_finite)
    return C

class EKF:
    """
    class for extended kalman filter providing (hopefully) user-friendly API
    """
    def __init__(self, nstate: int, nmeas: int, *, 
                    dtype: np.dtype = np.float64, delta: float = 1e-3):
        """Initialise EKF for a given dimension of system.
        """
        self._nstate  = nstate
        self._nmeas   = nmeas
        self._dtype   = dtype
        self._delta   = delta
        self._init_ss = False
        self._static_jacobian = None
        self._jacobian_func   = None
        self.x_k = np.zeros([self._nstate],self._dtype)
        self.P_k = np.zeros([self._nstate,self._nstate],self._dtype)
        self.K_k = np.zeros([self._nstate,self._nmeas],self._dtype)
        self.H_k = np.zeros([self._nmeas,self._nstate],self._dtype)

    def set_h_model(self,h_model,*args):
        """Set non-linear function h(x).
        h(x) must be a function of one vector argument, with length=nstate, and it must
        return a vector output, with length=nmeas.
        """
        self._h_model = h_model
        self._h_args  = args
        assert self._h_model(np.zeros(self._nstate,dtype=self._dtype),
                                *self._h_args).shape == (self._nmeas,)

    def update(self,e_k,use_static_jacobian=False):
        # perform upate
        self.P_k = self.SigV + self.Amat @ self.P_k @ self.Amat.T
        if use_static_jacobian:
            if self._static_jacobian is None:
                self._set_static_jacobian()
            self.H_k = self._static_jacobian
        else:
            self.H_k = self.linearise_h(self.x_k)
        self.K_k = np.linalg.solve(self.H_k @ self.P_k @ self.H_k.T + self.SigW,
                            self.H_k @ self.P_k.T).T
        self.x_k = self.Amat @ self.x_k + self.K_k @ e_k
        self.P_k = self.P_k - self.P_k @ self.H_k.T @ self.K_k.T    

    def update_ss(self,e_k):
        # perform steady state upate
        if not self._init_ss:
            self.init_steady_state()
        self.x_k = self.Amat @ self.x_k + self.K_ss @ e_k

    def init_steady_state(self,epochs=50):
        if self._static_jacobian is None:
            self._set_static_jacobian()
        self.H_ss = self._static_jacobian
        self.P_ss,KT_ss = self.DARE(self.Amat.T,self.H_k.T,self.SigV,self.SigW,epochs=epochs)
        self.K_ss = KT_ss.T
        self._init_ss = True
    
    def DARE(self,A,B,Q,R,epochs=50):
        alpha   = np.copy(A)
        beta    = np.copy(B@np.linalg.solve(R,B.T))
        gamma   = np.copy(Q)
        Id = np.eye(Q.shape[0])
        print("%d/%d Complete" % (0, epochs))
        old = alpha
        for it in range(epochs):
            common  = Id + beta@gamma
            gamma   = gamma + alpha.T@gamma@np.linalg.solve(common,alpha)
            beta    = beta + alpha@np.linalg.solve(common,beta)@alpha.T
            alpha   = alpha@np.linalg.solve(common,alpha)
            print("%d/%d Complete" % (it+1, epochs))
            diff = np.sum(np.abs(alpha.flatten()-old.flatten()))
            print("difference: %f" % diff)
            old     = alpha.copy()
        return [gamma,np.linalg.solve(R+B.T@gamma@B,B.T)@gamma@A]

    def _set_static_jacobian(self,x0=None):
        if x0 is None:
            x0 = 0.0*self.x_k
        self._static_jacobian = self.linearise_h(x0)

    def linearise_h(self,r):
        dr = np.zeros_like(r)
        dr[0] = self._delta
        y_0 = self._h_model(r,*self._h_args)
        y_len = y_0.shape[0]
        jacobian = np.zeros([y_len,dr.shape[0]],dtype=self._dtype)
        for j in range(dr.shape[0]):
            y_tmp = self._h_model(r+np.roll(dr,j),*self._h_args)
            jacobian[:,j] = (y_tmp-y_0) / self._delta
        return jacobian

    def set_matrices(self,Amat=None,SigV=None,SigW=None,SigW_inv=None):
        if Amat is not None:
            self.Amat = Amat.copy()
        if SigV is not None:
            self.SigV = SigV.copy()
        if SigW is not None:
            self.SigW = SigW.copy()
        if SigW_inv is not None:
            self.SigW_inv = SigW_inv.copy()

    def set_mats_from_scalars(self,*,alpha=0.995,sigsqx=1e-9,sigsqw=1e-1):
        Amat =  alpha * np.eye(self._nstate,dtype=self._dtype)
        SigX = sigsqx * np.eye(self._nstate,dtype=self._dtype)
        SigV = SigX - Amat @ SigX @ Amat.T
        SigW = sigsqw * np.eye(self._nmeas,dtype=self._dtype)
        SigW_inv = (1/sigsqw) * np.eye(self._nmeas,dtype=self._dtype)
        self.set_matrices(Amat=Amat,SigV=SigV,SigW=SigW,SigW_inv=SigW_inv)

    def reset(self):
        self.P_k = self.SigV.copy()
        self.H_k *= 0.0
        self.K_k *= 0.0
        self.x_k *= 0.0
        self._static_jacobian = None
    
    def set_jacobian_func(self,func):
        self._jacobian_func = func


class IteratedEKF(EKF):
    def update(self,meas_k):
        # perform upate
        self.P_k = self.SigV + self.Amat @ self.P_k @ self.Amat.T
        x_i = self.Amat @ self.x_k

        for it in range(10):
            #print(x_i)
            if self._jacobian_func is not None:
                H_i = self._jacobian_func(x_i)
            else:
                H_i = self.linearise_h(x_i)
            K_i = cho_solve(H_i @ self.P_k @ H_i.T + self.SigW,
                            H_i @ self.P_k.T).T
            x_i = self.x_k + K_i @ (meas_k-self._h_model(x_i)-H_i@(self.x_k-x_i))
        
        self.x_k = x_i 
        self.P_k = (np.eye(x_i.shape[0])-K_i @ H_i) @ self.P_k