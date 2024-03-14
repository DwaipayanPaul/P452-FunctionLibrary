# Library Module
import math
import csv
import random
import copy
import numpy as np
from matplotlib import pyplot as plt
#---------------NEW LIBRARY------------------

# ------------------Laplace equation---------------------
def laplace(A, mx):
 
    iterations = 0
    n=len(A)
    # Iterations
    while iterations < mx:
        Anew = A.copy()    
        for i in range(1,n-1):
            for x in range(1,n-1):
                # Update A
                A[i,x] = 0.25*(Anew[i,x+1]+Anew[i,x-1]+Anew[i+1,x]+Anew[i-1,x])
        iterations += 1




#--------------------Importance sampling---------

def monte_imp_sampling(a,b,n,f,g,ig):     # monte carlo method
    X=[]   # to store the random variables Xi
    w=[]   # weights
    for i in range(n):
        r=random.random()      # random number generator from [0,1]
        r1=a+((b-a)*r)         # r is converted to be in range [a,b]
        X.append(r1)           # storing in X
        
        w.append(g(r))
    # calculation of integral
    sum = 0.0
    for i in range(n):
        sum+=f(ig(X[i]))/g(ig(X[i]))      #summation of f(y(xi))/p(y(xi))
        p=((b-a)*sum)/n          # value of integral

    return p




# Guassian quadrature----------------------
def modified_func(func, y, a, b):     # conversion of the integration limit 
                                        # from [a,b] to [-1,1]             
    y2 = ((a+b)/2) + ((b-a)*y/2)
    return func(y2) * ((b-a)/2) 


def inte(func, l, u ,deg, s,w):
    
    # summation of w1f1 + w2f2 +....
    sum = 0
    for i in range(len(s)):
        sum += w[i] * modified_func(func, s[i], l, u)

    return sum


#-------------------HEAT WAVES---------------
def explicit_heat(init,a,b,k,dt,dx,nt = 100): 

    N = len(init)
    alp = (k*dt)/(dx**2)

    # for time steps nt
    for t in range(1,nt+1):
        new = np.zeros(N)
        # for general i anf j, value of u
        for i in range(1,N-1):
            new[i] = init[i] + alp*(init[i+1]+init[i-1]-2*init[i])

            # boundary value
            new[0] = a(t*dt)
            new[-1] = b(t*dt)

        # keeping the old config for the previous time
        if t==1 or t==11 or t==21 or t==51 or t==101 or t==201 or t==501:
            init = copy.deepcopy(new)
            x=np.linspace(0, N * dx,len(init))
            plt.plot(x,new,label='At temperature step = %s'%(t-1))
    plt.xlabel("X (units)")
    plt.ylabel("U(X) deg. C")
    plt.legend(loc='lower right',fontsize='small')
    plt.title("Temperature profile")
    plt.show()
    return new

def implicit_heat(init,a,b,k,dt,dx,nt = 50):

    N = len(init)
    alp = (k*dt)/(dx**2)

    # without boundary values
    ini = np.delete(init, [0,-1])
    new = np.zeros(N-2)

    # calculate u
    mat = np.zeros((N-2,N-2))
    for i in range(N-2):
        for j in range(N-2):
            if(i==j): mat[i][j] = 1+(2*alp)
            if(abs(i-j)==1): mat[i][j] = -alp
    
    I = [[(1 if i == j else 0) for j in range(N-2)] for i in range(N-2)]
    mat = inverse(mat, I)

    # time steps nt
    for t in range(nt):

        # Update the boundaries
        ini[0] = ini[0]+(alp*a(t*dt))
        ini[-1] = ini[-1]+(alp*b(t*dt))

        new = np.dot(mat,ini)

        ini = copy.deepcopy(new)
        
        x=np.linspace(0, 1,len(ini))
        plt.plot(x,ini)
        plt.ylim(0,1)
        plt.show()
    new = np.insert(new,0,a(nt*dt))
    new = np.append(new,b(nt*dt))

    return new
#---------------------------------------------


#------------------------------------------------
def modvec(x):    # mod of X vector: xT.x = |x|^2
    suma=0.0
    
    for i in range(len(x)):
        suma+=(x[i])*(x[i])
    return math.sqrt(suma)


def norm2d(x,y):   # finding the norm of (X - Y)
    sum1=0.0
    summ=0.0
    for i in range(len(x)):
        summ=x[i]-y[i]
        sum1+=abs(summ)   # summimng the mod of each elemnt
    return sum1

def matrix_mult(m,n):    # multiply two matrices - nxn and nx1
    l=len(m)
    r=[0 for i in range(l)]
    for i in range(l):
            for k in range(l):
                r[i] = r[i] + (m[i][k] * n[k])
    return r

def matsub(a,b):       #  substract matrices - nx1 and nx1
    r=[]
    for i in range(len(a)):
        r.append(a[i]-b[i])
    return r

def matadd(a,b):       #  add matrices - nx1 and nx1
    r=[]
    for i in range(len(a)):
        r.append(a[i]+b[i])
    return r

def innerprod(x,y):   # inner product of column matrices
    n=len(x)
    si=0.0
    for i in range(n):
        si+=x[i]*y[i]
    return si

def scalprod(c,x):   # scalar product: constant to a column matrix
    cx=[]
    for i in range(len(x)):
        cx.append(c*x[i])
    return cx
    
def transpo(a):
    b=[[0 for i in range(len(a[0]))] for j in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a)):
            b[j][i]=a[i][j]
    return b

# JACOBI Method (linear equations and inverse)
def jacobi(a,b,epsilon=10e-4):
    
    # guess the x
    x1,x2=[],[]      # two storages for Jacobi method
    '''print("Guess the x matrix of length",len(b))    # Guess Xo
    for i in range(len(b)):
        x1.append(float(input("element:")))'''
    x1=[3 for i in range(len(b))]     # guess matrix
    
    x2=[0 for i in range(len(x1))]            # initialising the other storage
    c=0   # to count the even and odd iteration 
    tarr=[]
    while norm2d(x1, x2)>epsilon:           # Iteration condition
        
        c+=1
        if c%2==0:                 # even iteration: x1 will store the x values
                                   # using x2 storage 
            for i in range(len(x1)):
                sums=0.0
                for j in range(len(x2)):
                    if i!=j:
                        sums+=a[i][j]*x2[j]
                x1[i]=(b[i]-sums)/a[i][i]   # calculation of X matrix
        else :                     # odd iteration: x2 will store the x values
                                   # using x1 storage
            
            for i in range(len(x2)):
                sums=0.0
                for j in range(len(x1)):
                    if i!=j:
                        sums+=a[i][j]*x1[j]
                x2[i]=(b[i]-sums)/a[i][i]   # calculation of x matrix
        tarr.append(norm2d(x1, x2))
    
    return x2,tarr


# Gauss - sidel method

def gauss_sidel(a,b,epsilon=10e-4):
    
    # guess the Xo
    x1=[]    # only one storage is required for gauss-sidel
    '''print("Guess the x matrix of length",len(b))
    for i in range(len(b)):
        x1.append(float(input("element:")))'''
        
    x1=[2 for i in range(len(b))]
    c=0    # no of iterations
    t=1    # to check with tolerence
    tarr=[]
    while t>epsilon:
        t=0.0 # tolerence
        t1=0.0
        c+=1
        
        for i in range(len(x1)):
            sums=0.0
            for j in range(len(x1)):
                if i!=j:
                    sums+=a[i][j]*x1[j]
            
            t1=x1[i]
            x1[i]=(b[i]-sums)/a[i][i]
            t+=abs(t1-x1[i])            # the change in new X and old X
        
        tarr.append(t)
    
    return x1,tarr



# Conjugate gradient method
def conjugate(a,b,e=10e-5):
    
    # guess xo
    x1=[]      # to store the X 
    '''print("Guess the x matrix of length",len(b))  # Guessing Xo
    for i in range(len(b)):
        x1.append(float(input("element:")))'''
    x1=[1 for i in range(len(b))]
    
    # calculate ro and do
    r=matsub(b,matrix_mult(a, x1))
    d=copy.deepcopy(r)
    i=0
    n=100      #len(a)
    tarr=[]
    while norm2d(r, [0 for i in range(len(r))]) > e and i<n:  # iteration condition
        
        al=innerprod(r,r)/innerprod(d, matrix_mult(a, d))      # calculating scaling factor
        
        x1=matadd(x1, scalprod(al, d))                         # calculating new X
        r2=matsub(r, scalprod(al, matrix_mult(a, d)))          # calculating new r
        
        # if the iteration continues
        bet=innerprod(r2, r2) /innerprod(r, r)            # calculating beta
        d=matadd(r2, scalprod(bet, d))             # new d
        r=copy.deepcopy(r2)                       # r old = r new
        i+=1
        tarr.append(norm2d(r, [0 for i in range(len(r))]))
    
    return x1,tarr

# Conjugate gradient on fly--------

def matrix_mult_fly(f,n):    # multiply two matrices - nxn and nx1
    l=len(n)
    N=math.sqrt(l)
    r=[0 for i in range(l)]
    for i in range(l):
            for k in range(l):
                r[i] = r[i] + (f(i,k,N) * n[k])
    return r

def conjugate_fly(f,b,e=10e-5):
    
    # guess xo
    x1=[]      # to store the X 
    '''print("Guess the x matrix of length",len(b))  # Guessing Xo
    for i in range(len(b)):
        x1.append(float(input("element:")))'''
    x1=[1 for i in range(len(b))]
    
    # calculate ro and do
    r=matsub(b,matrix_mult_fly(f, x1))
    d=copy.deepcopy(r)
    i=0
    n=len(b)
    tarr=[]
    tarr.append(norm2d(r, [0 for i in range(len(r))]))
    while norm2d(r, [0 for i in range(len(r))]) > e and i<n:  # iteration condition
        
        al=innerprod(r,r)/innerprod(d, matrix_mult_fly(f, d))      # calculating scaling factor
        
        x1=matadd(x1, scalprod(al, d))                         # calculating new X
        r2=matsub(r, scalprod(al, matrix_mult_fly(f, d)))          # calculating new r
        
        # if the iteration continues
        bet=innerprod(r2, r2) /innerprod(r, r)            # calculating beta
        d=matadd(r2, scalprod(bet, d))             # new d
        r=copy.deepcopy(r2)                       # r old = r new
        i+=1
        tarr.append(norm2d(r, [0 for i in range(len(r))]))
    
    return x1,tarr

#---------------------------------



# power-method for calculating dominant eigenvalues (e: tolerence)
def powermet(a,co,e=10e-5,ll=[],v1=[]):     # a is matrix, co is no of eigenvalues requested
    x=[]                                    # ll and v1 are for storing eigenvalues and eigenvectors
    n=len(a)
    print("Guess the x matrix of length",n)    # Guess a matrix of length n
    for i in range(n):
        x.append(float(input("element:")))
    t=1.0
    l=1.0                          # initialise tolerence and old lambda
    x1=[0.0 for i in range(n)]   # initialise old eigen vector
    # if requested no of eigenvalue is more than matrix size
    if co>n:
        print("No of eigenvalue requested is more than the size of matrix")
        return
    
    while(t>e):
        
        # multiply, new x = A * old x
        for i in range(n):
            c=0
            for j in range(n):
                c+=a[i][j]*x[j]
            x1[i]=c
        
        
        # copy matrices new x to old x
        for i in range(n):
            x[i]=x1[i]
        
        # finding the max in x
        l1=abs(x[0])
        for i in range(1,n):
            if abs(x[i])>l1:
                l1=abs(x[i])
        
        # dividing x by max element
        for i in range(n):
            x[i]=x[i]/l1
        
        # difference between old and new lambda
        t=abs(l1-l)
        
        if t>e:
            l=l1    # if diff is more than tolerence, then old lambda=new lambda
                    # the iteration continues
        else:
            print('Eigenvalue and corresponding eigenvector:')
            print(l1,x)     # print eigen value and corresponding eigenvector
            print()
            # storing eigenvalues and eigenvectors
            ll.append(l1)
            v1.append(x)
            co-=1        # decrease eigenvalue no. counter
            u=[]         # calculating normalised eigenvector U
            for i in range(n):         
                u.append(x[i]/modvec(x))
            
            for i in range(n):        # calculating A* = A - lamda1 * U1 * transp(U)
                for j in range(n):
                    a[i][j]=a[i][j]-(l1*u[i]*u[j])
                    
            
            if co>=1:                # continue finding next eigenvalue, if more than 1 is requested
                print("Next Eigenvalue---")
                powermet(a,co,e,ll,v1)
            return ll,v1

# Jacobi method for finding eigenvalues-----------------------------------
def largoff(a):   # finding largest non-diagonal element in A
    n=len(a)
    p,q=0,0
    t=a[0][1]
    for i in range(n):
        for j in range(n):
            if i<j and abs(a[i][j])>=abs(t):
                t=a[i][j]
                p,q=i,j
    return t,p,q

def check(a):     # To check is A is a diagonal matrix
    n=len(a)
    for i in range(n):
        for j in range(n):
            if i!=j and abs(a[i][j])>10e-6:
                
                return True
                
    return False

def jacobeig(a):    # Jacobi Method
    n=len(a)
    c,p,q=largoff(a)
    while check(a)==True :

        try:    # calculating thetha in Givens rotation matrix S
            th = math.atan((2 * a[p][q]) / (a[q][q] - a[p][p])) / 2
            
        except ZeroDivisionError:   # handling division by zero error
            th=math.pi/4
            
        # Calculating the B = s^TAs, and storing it in A itself
        p1 = a[p][p]    # temporary terms 
        p2 = a[q][q]    # which are required to be replaced, but used later
        a[p][p] =p1*(math.cos(th)**2) + p2*(math.sin(th)**2) - 2*a[p][q]*math.cos(th)*math.sin(th)
        a[q][q] =p2*(math.cos(th)**2) + p1*(math.sin(th)**2) + 2*a[q][p]*math.cos(th)*math.sin(th)
        a[p][q] =(p1-p2)*math.cos(th)*math.sin(th) + a[p][q]*(math.cos(th)**2 - math.sin(th)**2)
        a[q][p] = a[p][q]
        for i in range(n):
            p1 = a[i][p]  # temporary terms
            p2 = a[i][q]
            if i != p and i != q:
                a[i][p] = p1*math.cos(th) - p2*math.sin(th)
                a[p][i] = a[i][p]
                a[i][q] = p2*math.cos(th) + p1*math.sin(th)
                a[q][i] = a[i][q]
        
        # if the iteration continues        
        c,p,q=largoff(a)        
        
    u=[]   # to store the diagonal elements which are the eigen values
    for i in range(n):
        u.append(a[i][i])
    return u
# ----------------------------------------------------------------------------       

# Jack-knife (and bootstrap)-------------------------------------------------
def mean(x):  # to find mean of a data set
    n=len(x)
    s=0.0
    for i in range(n):
        s+=x[i]
    return s/n

def vari(x):    # to find variance of dataset
    m=mean(x)
    n=len(x)
    s1=0.0
    for i in range(n):
        s1+=(x[i]-m)**2
    return s1/(n)
        

def jackknife(x,f):    # Jack knife estimation of a function: f
    n=len(x)
    mn=0.0
    u=[]
    u1=[]
    for i in range(n):  # Taking jackknife samples and storing in u
        u=copy.deepcopy(x)
        u.pop(i)
        u1.append(f(u))
        
    return mean(u1),vari(u1)        # return the mean and variance of the estimation


def bootstrap(x,b,f):   # bootstrap estimation: x dataset, b iterations, function f 
    u=[]
    n=len(x)            
    ba=[i for i in range(b)]
    y=[]
    sb=0.0
    for i in range(b):
        for j in range(n):     # bootstrap sampling, b times
            u.append(random.choice(x))
        y.append(f(u))
        sb+=f(u)            
    plt.plot(ba,y)  
    plt.title('plot')
    plt.show()
    return sb/b           # returining the mean of function estimates 
#-------------------------------------------------------

#----Chi-square fit----------------------------

# linear regression-------------
def f(x,a,b):
    return a+b *x      # linear equation, a and b are parameters
    
def param(x,y,sig,f=f,dof=2,stx='X',sty='Y'):  # x, y dataset; sig is uncertanity; f is linear function; dof is degrees of freedom
    n=len(x)
    s,sx,sxx,syy,sy,sxy=0.0,0.0,0.0,0.0,0.0,0.0
    for i in range(n):
        s+=(1/sig[i])**2
        sx+=x[i]/sig[i]**2
        sy+=y[i]/sig[i]**2
        sxx+=(x[i]/sig[i])**2
        syy+=(y[i]/sig[i])**2
        sxy+=(x[i]*y[i])/sig[i]**2
    
    # finding del
    d=(s*sxx)-(sx**2)
    # estimating parameters
    a=(sxx*sy - sx*sxy)/d
    b=(s*sxy - sx*sy)/d
    # variance in parameters
    siga=sxx/d
    sigb=s/d
    # cov(a,b) and r^2
    cov=-sx/d
    rsq=sxy/(sxx*syy)
    
    sum1=0.0
    # chi-square calculation
    for i in range(n):
        sum1+=(y[i]-f(x[i],a,b)/sig[i])**2
    yd=[]
    for i in range(n):
        yd.append(f(x[i],a,b))
    # plotting the fitted line
    plt.plot(x,y,label='Observed data')
    plt.errorbar(x, y, yerr = sig,fmt='o',ecolor = 'red',color='yellow')
    plt.plot(x,yd,label='Fitted data, f = %.2f + %.2f x' %(a, b))
    plt.title('Linear regression, f=a+bx')
    plt.xlabel(stx)
    plt.ylabel(sty)
    plt.legend()
    plt.show()    
    return a,b,siga,sigb,cov,rsq,sum1/dof  # returning parameters a,b, variances,cov,r^2
                                            # and reduced chi-square

# Polynomial model --------------------------------
def ff(x,deg,par):   # polynomial function: x dataset, deg degree of polynomial, par paramters array
    su11=0.0
    for i in range(deg+1):
        su11+=par[i]*(x**i)
    return su11
    
def poly(x, y, sig, deg,chebys=False):  # fiting polynomial model: x,y dataset; sig is uncertainity; deg is degree
    n = len(x)
    # intialising matrices for formulation
    A = [[0 for j in range(deg + 1)] for i in range(deg + 1)]
    b = [0 for i in range(deg + 1)]
    
    for i in range(n):     # populating A and b, Ax=b where x is params
        for j in range(deg + 1):
            b[j] += (x[i] ** j) * y[i] / (sig[i] ** 2)
            for k in range(deg + 1):
                A[j][k] += (x[i] ** (j + k)) / (sig[i] ** 2)
    cond = np.linalg.cond(A)     # condition number
    
    lu(A)
    params = forward_backward(A, b)   # parameters a0,a1,a2....
    
    chi = 0
    for i in range(n):    # calculating chi-square
        sum = 0
        for j in range(deg + 1):
            sum += params[j] * math.pow(x[i], j)
                
        chi += math.pow((y[i] - sum), 2)/sum
    
    I = [[(1 if i == j else 0) for j in range(deg + 1)] for i in range(deg + 1)]
    
    A_inv = inverse(A, I)     # covariance of the paramters
    
    # plot the fitted polynomial
    plt.plot(x,y,'bo',label="Observed data")
    u=np.linspace(min(x),max(x),100)
    plt.plot(u,ff(u,deg,params),label="Normal fitted data")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return params, A_inv, chi/(n-deg),cond    # returning paramter array, covariance
                                        # reduced chi-square and condition number
                                        
                                        
def poly_mod(x, y, sig, deg, mod_bas):  # fiting modified polynomial model    

    n = len(x)
    # intialising matrices for formulation
    A = [[0 for j in range(deg + 1)] for i in range(deg + 1)]
    b = [0 for i in range(deg + 1)]
    
    
    for i in range(n):     # populating A and b, Ax=b where x is params
        for j in range(deg + 1):
            
            b[j] += mod_bas(j,x[i]) * y[i] / (sig[i] ** 2)
            for k in range(deg + 1):
                A[j][k] += ((mod_bas(j,x[i]))*(mod_bas(k,x[i]))) / (sig[i] ** 2)
    cond = np.linalg.cond(A)         # condition number
                
    lu(A)
    params = forward_backward(A, b)   # parameters a0,a1,a2....
    
    chi = 0
    for i in range(n):    # calculating chi-square
        sum = 0
        for j in range(deg + 1):
            sum += params[j] * math.pow(x[i], j)
                
        chi += math.pow((y[i] - sum), 2)/sum
    
    I = [[(1 if i == j else 0) for j in range(deg + 1)] for i in range(deg + 1)]
    
    A_inv = inverse(A, I)     # covariance of the paramters
    
    # plot the fitted polynomial
    plt.plot(x,y,'bo',label="Observed data")
    yfit =[]
    for i in range(len(x)):
        summ=0
        for j in range(deg):
            summ+=params[j]*mod_bas(j, x[i])
        yfit.append(summ)

    # plot the fitted polynomial
    #plt.plot(x,y,'bo',label="Observed data")
    #u=np.linspace(min(x),max(x),100)
    
    plt.plot(x,yfit,label="Modified fitted data, degree=%s"%deg)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
    return params, A_inv, chi/(n-deg),cond    # returning paramter array, covariance
                                        # reduced chi-square and condition number

#---------------RANDOM NUMBER---------------------------
def mlcg(a,m,N,seed=69):  # random number generator
    x=[0 for i in range(N)]
    x[0]=seed
    for i in range(1,N):
        x[i]=(a*x[i-1])%m     # random between 0 and m
        x[i-1]=x[i-1]/m       # number between 0 and 1
    x[-1]=x[-1]/m            # changing the last number
    return x

def monte_carlo_mod(a,b,f,X):     # monte carlo method: a,b are limits,f is the integral and X is random no
                                                                    # from 0 to 1
    n=len(X)
    for i in range(n):
        r=X[i]                   # random number generator from [0,1]
        X[i]=a+((b-a)*r)         # r is converted to be in range [a,b]

        
    # calculation of integral
    sum = 0.0
    for i in range(n):
        sum+=f(X[i])        # f(x1) + f(x2) +.... f(xN)
    p=((b-a)*sum)/n          # value of integral

    return p

# throwing point method to find pi
def throw_pi(X,Y,rr = 1):     # x,y random number arrays, radius rr of the circle
    n=len(X) 
    
    ini=0
    X = [(r-0.5)*2*rr for r in X]           # converting to (x,y) in a square of side 2 units
    Y = [(r-0.5)*2*rr for r in Y]
    
    for i in range(n):
        
        if (X[i]**2 + Y[i]**2)<=(rr**2):
            ini+=1
    
    return 4*(ini/n)

 

#---------------------------------------------------------------------------

# ---------------OLD LIBRARY-------------------------------------------------
############## ASSIGNMENT 7 ###################
# Euler's method
def eulerm(x0,y0,st,h,f,N):
    x,y=[],[]     # to save the data for x and y(x)
    # initial values
    y.append(y0)
    x.append(x0)
    # N=1000
    for i in range(1,N):
        z=y[i-1]+(h*f(y[i-1],x[i-1]))  # y(x_n + h) = y(x_n) + h f(y(x_n), x_n)
        y.append(z)    # storing y(x_n + h)
        x.append(x0+(i*h))  # and x
    write_csv(st, y, x)     # writing the arrays in a csv file

def runge_kutta(f1,f2,x0,y0,z0,h,N,st):    # Runge-Kutta 4
    ax,ay=[],[]   # arrays to store x and y(x) respectively
    # stores the initial value
    ax.append(x0)
    ay.append(y0)
    z=z0

    for i in range(N):   # values of k1,k2,k3,k4 for f1 and f2
        #k1
        k1=f1(z,ay[i],ax[i])
        k1t=f2(z,ay[i],ax[i])
        #k2
        k2=f1(z+(k1t*h)/2, ay[i]+(k1*h)/2, ax[i]+h/2)
        k2t=f2(z+(k1t*h)/2, ay[i]+(k1*h)/2, ax[i]+h/2)
        #k3
        k3=f1(z+(k2t*h)/2, ay[i]+(k2*h)/2, ax[i]+h/2)
        k3t=f2(z+(k2t*h)/2, ay[i]+(k2*h)/2, ax[i]+h/2)
        #k4
        k4=f1(z+(k3t*h), ay[i]+(k3*h), ax[i]+h)
        k4t=f2(z+(k3t*h), ay[i]+(k3*h), ax[i]+h)
        # z and y
        z=z+((k1t+2*k2t+2*k3t+k4t)*h)/6
        yp=ay[i]+((k1+2*k2+2*k3+k4)*h)/6
        # storing x and y in ax and ay
        ay.append(yp)
        xp=x0+(h*(i+1))
        ax.append(xp)

    write_csv(st, ay, ax)  # writing x and y(x) in a csv file

    return yp,ax,ay      # returning y(x=xn)

def shooting_method(f1,f2,x0,y0,yn,h,N,st,g):
    
    ax,ay =[],[]
    y,ax,ay = runge_kutta(f1, f2, x0, y0, g, h, N, "test.csv")    # yn for the first guess
    print(f"Value of y(x=xn) for the above guess {g}=",y)

    # initialising
    lower,upper=0.0,0.0
    # checking if y overshoots or undershoots
    # if y overshoots
    if y > yn and abs(y - yn) > 10E-4:
        upper = g
        # we got upper bracket of y
        # to find lower bound
        while y > yn:
            g = float(input(f"Guess a value of y\'({x0}) lower than the previous guess\n"))
            y,ax,ay  = runge_kutta(f1, f2, x0, y0, g, h, N, "test.csv")
            print(f"Value of y(x=xn) for the above guess {g}=", y)

        if abs(y - yn) < 10E-4:                # if yn for the guess is equal to or very near to actual yn
            y,ax,ay = runge_kutta(f1, f2, x0, y0, g, h, N, st)     # writing the final data file st
            print("Value of y(x=xn) found, integration successful")
            return y,ax,ay
        else:                                # if yn of guess is less than actual yn
            lower = g                        # then we have found the lower bracket
            g1=lagrange_interpolation(upper,lower,f1,f2,x0,y0,yn,h,N,st)

        # if y undershoots
    elif y < yn and abs(y - yn) > 10E-4:
        lower = g     # got the lower bracket
        # now to find upper bound
        while y < yn:
            g = float(input(f"Guess a value of y\'({x0}) greater than the previous guess\n"))
            y,ax,ay = runge_kutta(f1, f2, x0, y0, g, h, N, "test.csv")
            print(f"Value of y(x=xn) for the above guess {g}=", y)

        if abs(y - yn) < 10E-4:           # if yn for the guess is equal to or very near to actual yn
            y,ax,ay = runge_kutta(f1, f2, x0, y0, g, h, N, st)      # writing the final data file st
            print("Value of y(x=xn) found, integration successful")
            return y,ax,ay
        else:
            upper = g
            g1=lagrange_interpolation(upper, lower, f1, f2, x0, y0, yn, h, N, st)
        #
    elif abs(y - yn) < 10E-4:           # if yn for the guess is equal to or very near to actual yn
        y,ax,ay = runge_kutta(f1, f2, x0, y0, g, h, N, st)  # if guess gives perfect value of yn at xn, then solution is obtained
        print("Value of y(x=xn) found, integration successful")
        return y,ax,ay

    return g1,ax,ay

def lagrange_interpolation(upper,lower,f1, f2, x0, y0, yn, h, N, st):
    yl,ax,ay = runge_kutta(f1, f2, x0, y0, lower, h, N, st)    # yn for lower bracket

    yh,ax,ay = runge_kutta(f1, f2, x0, y0, upper, h, N, st)    # yn for upper bracket
    # for next y'(x0)
    g = lower + ((upper - lower) / (yh - yl)) * (yn - yl)

    y,ax,ay = runge_kutta(f1, f2, x0, y0, g, h, N, st)   # yn for the new y'(x0)
    print("Value of y(x=xn) found, integration successful",g)
    return g,ax,ay


###################### Assignment 6  #########################################################
# Methods for numerical integration
def midpoint(a,b,n,f):     # Midpoint/Rectangle method
    # width of N equal parts
    h=(b-a)/n
    # x stores values of x1, x2, x3........
    x=0.0
    sum =0.0
    for i in range(n): # values of x
        x=((a+i*h)+(a+((i+1)*h)))/2
        sum += h * f(x)    # summing the values of h*f(x1),h*f(x2),.....

    return sum

def trapezoidal(a,b,n,f):    # Trapezoidal method
    h = (b - a) / n
    # x stores values of x0, x1, x2, x3........ xN
    x = 0
    sum = 0
    w = 1  # intialising weight function w=1 or 2
    for i in range(n+1):
        x = a + (i * h)   # values of x
        if i==0 or i==n:   # w(x0)=w(xN)=1
            w=1
        else: w=2          # w(x1)=w(x2)=.....w(x{N-1})=2
        sum += h * w * f(x)/2  # summing values of h * w * f(xi)/2

    return(sum)

def simpsons(a,b,n,f):    # simpsons method
    h = (b - a) / n

    x = [0 for i in range(n + 1)]   # x stores x0, x1, x2.....
    for i in range(0,n+1,2):
        x[i] = a + (i * h)      # putting the values of x0, x2, x4,....x2n

    for i in range(1,n,2):      # putting the values of x1,x3,x5....
        x[i]=(x[i-1]+x[i+1])/2  # x1 is avg of x0 and x2, x3 is avg of x2 and x4 and so on..
    sum = 0
    w = 1  # intialising weight function w=1,2 or 4

    for i in range(n + 1):
        if i == 0 or i == n:  # w(x0)=w(xN)=1
            w = 1
        elif i%2==0:         # w(xi)=2 for even i
            w = 2
        else: w=4             # w(xi)=4 for odd i
        sum += h * w * f(x[i]) / 3    # summing values of h * w * f(xi)/3

    return sum

def monte_carlo(a,b,n,f):     # monte carlo method
    X=[]   # to store the random variables Xi

    for i in range(n):
        r=random.random()      # random number generator from [0,1]
        r1=a+((b-a)*r)         # r is converted to be in range [a,b]
        X.append(r1)           # storing in X

    # calculation of integral
    sum = 0.0
    for i in range(n):
        sum+=f(X[i])        # f(x1) + f(x2) +.... f(xN)

    p=((b-a)*sum)/n          # value of integral

    return p




def mperr(a, b, e, fd2):        # Error for midpoint method, gives the value of N
    N = float(((((b - a) ** 3) * fd2) / (24 * e)) ** (1 / 2))
    return math.ceil(N)           # returns smallest integer bigger than N


def tperr(a, b, e, fd2):        # Error for trapezoidal method, gives the value of N
    N = float(((((b - a) ** 3) * fd2) / (12 * e)) ** (1 / 2))
    return math.ceil(N)        # returns smallest integer bigger than N


def sperr(a, b, e, fd4):        # Error for simpsons method, gives the value of N
    N = float(((((b - a) ** 5) * fd4) / (180 * e)) ** (1 / 4))
    if N % 2.0 == 0.0:         # only returns even smallest integer bigger than N
        return math.ceil(N)
    else:
        return math.ceil(N + 1)

# To write an array in a csv file
def write_csv(str,er,er1):
    with open(str, 'w', newline='') as file:  # str: the name of file
        writer = csv.writer(file)
        writer.writerow(["X", "y(x)"])  # The first row of the file
        for i in range(len(er)):
            writer.writerow([er1[i], er[i]])     # No of iteration in first column and absolute error in second column

#################################################################################################################
#
def funcdt(func,x):    # deritive of function
    h = 0.0001
    y = (func(x + h) - func(x-h)) / (2*h)
    return y

def bracket(func,a,b):  # bracketing of the root


    for i in range(12):   # iteration limit: 12

        if func(a)*func(b)<0: # roots are on either sides of root(bracketing done)
            print("Bracketing complete: a=",a,"and b=",b)
            return a,b

        elif func(a)*func(b)>0:  # roots are on same side w.r.t.the root

            if abs(func(a))<abs(func(b)):  # need to shift a
                a-=1.5*(b-a)

            elif abs(func(a))>abs(func(b)): # need to shift b
                b=b+1.5*(b-a)

def bisecting(func,a,b):    # Bisection method
    i=0             # counter for iterations
    error=[]         # stores the error values for each iterations
    print("Bisecting method:")
    while(b-a>1E-6 and i<=100):    # iteration limit: 100 and abs error limit: 1E-6

        error.append(round((b - a), 7))  # error value gets appended in the array

        c=(a+b)/2             # midpoint of a and b
        if func(a)*func(c)<0:   #   if a and c are on either sides of root
            b=c               #  b is shifted to c
        elif func(a)*func(c)>0: # if a and c are on same sides of root
            a=c                # a is shifted to c
        else:
            print("Solution found:",c)   # if c is the root:f(c)=0
            return 0
        i+=1
    # print solution
    print("The solution lies between a=",a,"b=",b)
    return error  # return the error array

def falsi(func,a,b):      # Regular falsi method
    i=0            # counter for iterations
    error=[]       # stores the error values for each iterations
    print()
    print("Regular Falsi method:")

    x1,x2=a,b   # counters for calculating error: C_i+1-C_i
    while (abs(x2 - x1) > 1E-6 and i <= 200):   # iteration limit: 200 and abs error limit: 1E-6

        error.append(round(abs(x2 - x1), 7))     # error value gets appended in the array
        # False root c
        c = b - ((b - a) * func(b) / (func(b) - func(a)))

        if func(a) * func(c) < 0:  #   if a and c are on either sides of root
            b = c               #  b is shifted to c
        elif func(a) * func(c) > 0:   # if a and c are on same sides of root
            a = c              # a is shifted to c
        else:
            print("Solution found:", c)   # if c is the root:f(c)=0
            return 0

        if i%2==0:     # if the iteration no. is even
            x2=c       # C_2n=c
        else:
            x1=c       # else C_2n+1=c
        i += 1
    # print output
    print("The solution lies in range ",x1,"and ",x2)
    return error    # return the error array

def newton(func,x1):      # Newton-Raphson method
    i=0         # counter for iterations
    error=[]     # stores the error values for each iterations
    x2=0      # x1 and x2 are counters for calculating error: X_i+1-X_i
    print()
    print("Newton-Raphson method:")

    while(abs(x2-x1)>1E-4 and i<=200):    # iteration limit: 200 and abs error limit: 1E-6

        error.append(round(abs(x2 - x1), 7))    # error value gets appended in the array

        if i%2==0:       # if the iteration no. is even
            x2=x1-(func(x1)/funcdt(func,x1))   # X_2n=X_2n+1 -[f(X_2n)/f'(X_2n+1)]
        else:                                 # else
            x1=x2-(func(x2)/funcdt(func,x2))    # X_2n+1=X_2n -[f(X_2n+1)/f'(X_2n)]
        i+=1
    # print the solution
    print("The solution lies in range ", x2, "and", x1)
    return error,x1    # return the error array

# For Q2
def funct(x,a):  # function to calculate f(x) of a polynomial for a value x
                 # a is the array of co-efficients starting with the constant
    n=len(a)
    sum=0.0
    #  a[i] corresponds to the coefficient of x^i
    for i in range(n-1,-1,-1):
        sum+=a[i]*(x**i)   # stores the value of f(x)
    return sum

def functd1(x,a):  # 1st order derivative of a function (in case of polynomial)
    h=0.001
    y=(funct(x+h,a)-funct(x-h,a))/(2*h)    # f'(x)=[f(x+h)-f(x)]/2h
    return y

def functd2(x,a):  # 2nd order derivative of a function (in case of polynomial)
    h = 0.001
    y = (funct(x + h, a) + funct(x - h, a)-2*funct(x,a)) / (h*h)  # f"(x)=[f(x+h)-f(x-h)]/h^2
    return y

def deflate(sol,a):  # deflation
                     # a is the array of co-efficients starting with the constant
    n=len(a)
    q=[0 for i in range(n-1)]  # intialization of q(x)=p(x)/(x-x_o)
    q[n-2]=a[n-1]      # coefficient of x^n in p is x^n-1 in q
    # synthetic division
    for i in range(n-3,-1,-1):  # from x^n-2 in q
        q[i]=a[i+1]+(sol*q[i+1])

    return q   # final q

def solut(a,i): # to find solutions: i is the guess
                # a is the array of co-efficients starting with the constant
    n=len(a)  # n-1 is no. of roots

    if n!=2:  # when f is not of form: f(x)=x-c

        j1,j2=i,0  # counters for error: \alpha_i+1-\alpha_i
        j = i  # takes the guess for the solution
        a1=0   # a1 is for calculation of a=n/[G(+-)math.sqrt((n-1)*(nH-G^2))]
        k=1   # counter for iterations
        if funct(i,a)!=0:  # when i is not the root of f(x)

            while abs(j2-j1)>1E-6 and k<200:  # iteration limit: 200 and abs error limit: 1E-6
                # calculation G and H
                g=functd1(j,a)/funct(j,a)
                h=g**2-(functd2(j,a)/funct(j,a))
                # denominators : d1 and d2
                det1=g+math.sqrt((n-1)*(n*h-g**2))
                det2=g-math.sqrt((n-1)*(n*h-g**2))

                if abs(det1)>abs(det2):  # if absolute value of det1 is max
                    a1=n/det1          # a=n/[G(+)math.sqrt((n-1)*(nH-G^2))]
                else:
                    a1=n/det2          # a=n/[G(-)math.sqrt((n-1)*(nH-G^2))]

                if k%2==0:          # for even no. iteration
                    j1=j2-a1         # \alpha_2n+1=\alpha_2n - a
                    j=j1            # for next iteration: \alpha_2n+1
                            # else
                else:
                    j2=j1-a1        # \alpha_2n=\alpha_2n+1 - a
                    j=j2            # for next iteration: \alpha_2n
                k+=1

        # The iteration ended in even no.
        if k%2==0:
            print(j1)   # \alpha_2n+1 is the nearest solution
            # deflation and saving the new polynomial q as a(j1 is solution)
            a=deflate(j1,a)
        else:          # else
            print(j2)   # \alpha_2n is the nearest solution
            # deflation and saving the new polynomial q as a(j2 is solution)
            a = deflate(j2, a)
        # return the new polynomial array a
        return a

    else:  # when f is of form: f(x)=x-c
        if a[1]*a[0]<0 or a[1]<0:  # if eq is of form: x-c=0 or -x+c=0
            print(a[0]) # print         x=c (solution)
        else:                    # if eq is of form: -x+c=0
            print(-a[0]) # print        x=-c (solution)

        return 0


# THE LIBRARY MODULE ( with functions involving matrices)

def read_write(st):        # reading and writing matrix
    a=[]
    # Reading matrices from the files
    f1 = open(st, 'r')
    for line in f1.readlines():
        a.append([float(x) for x in line.split()])  # adding rows
    return a

def part_pivot(a,b):      # partial pivoting
    n=len(a)
    # initialise
    (c,d)=(0,0)
    for k in range(n-1):
        if a[k][k]==0:     # checking if the diagonal element is zero
            for r in range(k+1,n):
                if abs(a[r][k])>abs(a[k][k]):   # swapping
                    for i in range(n):
                        # swapping in matrix b
                        c = b[r]
                        b[r] = b[k]
                        b[k] = c
                        # swapping in matrix a
                        d=a[k][i]
                        a[k][i]=a[r][i]
                        a[r][i]=d

def part2d(a,b):    # partial pivoting when b is 2d matrix
    n=len(a)
    # initialise
    (c,d)=(0,0)
    for k in range(n-1):
        if a[k][k]==0:     # checking if the diagonal element is zero
            for r in range(k+1,n):
                if abs(a[r][k])>abs(a[k][k]):   # swapping
                    for i in range(n):
                        # swapping in matrix b
                        c = b[r][i]
                        b[r][i] = b[k][i]
                        b[k][i] = c
                        # swapping in matrix a
                        d=a[k][i]
                        a[k][i]=a[r][i]
                        a[r][i]=d
    

def lu(a):

    n=len(a)
    
    for j in range(n):       # keeping the column fixed
        for i in range(n):
            (sum1, sum2) = (0.0, 0.0)   # initialise
            # Condition for upper matrix (i<j and i=j)
            if i<=j:
                for k in range(i):
                    sum1+=a[i][k]*a[k][j]
                # overwriting a[i][j] as Uij
                a[i][j]=a[i][j]-sum1
            # Condition for lower matrix (i>j)
            elif i>j:
                for k in range(j):
                    sum2+=a[i][k]*a[k][j]
                # overwriting a[i][j] as Lij
                a[i][j]=(a[i][j]-sum2)/a[j][j]


    return a

# forward and backward substitution
# LUX=B  (here, B is each column of identity)
def forward_backward(a,b):
    n=len(a)
    # initializing X and Y matrix
    x = [0.0 for i in range(n)]
    y = [0.0 for i in range(n)]

    for i in range(n):
        # initialise
        (sumf,c1)=(0.0,0.0)   # c1 will act like an element in L
        # forward substitution: LY=B
        for j in range(i):
            if i==j:       # At diagonal, L[i][i]=1
                c1=1       # hence c1=1
            elif i>j:      # Below the diagonal, it is same as A
                c1=a[i][j]
            # (else) Above the diagonal, elements are zero
            #         c1=0
            sumf+=c1*y[j]
        # storing in matrix Y
        y[i]=b[i]-sumf

        # backward substitution: UX=Y
    for i in range(n-1,-1,-1):  # from n-1 to 0 (backwards)
        # initialise
        (sumb,c2)=(0.0,0.0)   # c2 will act like an element in U

        for k in range(n-1,i,-1): # from n-1 to i+1 (backwards)
            if i<=k:          # at diagonal and above it, U is same as A
                c2=a[i][k]
                # (else) below the diagonal, elements are zero
                #        c2=0
            sumb+=c2*x[k]
        # Storing in the matrix X (which are the columns of the inverse of A)
        x[i]=(y[i]-sumb)/a[i][i]
        
    return x

#------------------cholesky------------------
def cholesky(a):
    n=len(a)
    l=[[0 for i in range(n)] for j in range(n)]
    for i in range(n):       # keeping the column fixed
        for j in range(i+1):
            sum1=0.0   # initialise
            # Condition for upper matrix (i<j and i=j)
            for k in range(j):
                sum1+=l[i][k]*l[j][k]
            if i!=j:
                l[i][j]=(a[i][j]-sum1)/l[j][j]

            else:
                l[i][i]=math.sqrt(a[i][i]-sum1)
    
    return l

# forward and backward substitution for cholesky

def forward_backward_ch(l,b):
    n = len(b)
    #Forward Substitution

    for i in range(n):
        sum = 0
        for k in range(i):
            sum += l[i][k]*b[k]
        b[i] = (b[i] -  sum)/l[i][i]

    #Backward Substitution

    for i in range(n-1,-1,-1):
        sum = 0
        for k in range(i+1,n):
            sum += l[k][i]*b[k]
        b[i] = (b[i] - sum)/l[i][i]

    return b


def checkinv(a):
    tr=0.0
    for i in range(len(a)):
        tr+=a[i][i]
    
    return tr


def inverse(a,I):
    x=lu(a)           # L U decomposition
    n=len(a)
    det=1             # stores determinant of A
    # det(A)=det(U)=product of Uii for i in range(n)
    for i in range(n):
        det*=a[i][i]       # Aii=Uii
    # checking if inverse exist
    if det==0:
        print("Inverse of A does not exist.")
    else:
        #print("Inverse of A exists. It determinant:",det)
        print()
        # initialise
        z=[[ 0.0 for i in range(n) ] for j in range(n)]
        z1=[[ 0.0 for i in range(n) ] for j in range(n)]
    
        # Finding inverse:
        # Each column of identity(b) is B
        # Ax=B:   Forward-Backward substitution is done for each column of b(identity)
        for i in range(n):
            # [row[i] for row in b] is the ith column in b(identity)
            z[i]=forward_backward(a,[row[i] for row in I])  # z stores the x columns in each row
            for j in range(n):
                z1[j][i]=z[i][j]      # z1 stores them as column from z
    
        # z1 is the inverse
        # print
        return z1

def matrix_multp(m,n):    # multiply two matrices
    l=len(m)
    r=[[ 0.0 for i in range(l) ] for j in range(l)]
    for i in range(l):
        for j in range(l):
            for k in range(l):
                r[i][j] = r[i][j] + (m[i][k] * n[k][j])
    return r

def print_mat(a):      # print a matrix
    n=len(a)
    for i in range(n):
        for j in range(n):
            print(a[i][j]," ",end="")
        print()


def Gaus_Jordan(a,b):
    n = len(a)
    # partial pivoting before applying the method
    part_pivot(a,b)
    for k in range(n):
        pivot = a[k][k]
        # dividing by the pivot (for the diagonal elements)
        for j in range(k, n):
            a[k][j] = a[k][j] / pivot
        
        b[k] = b[k] / pivot
        # Transforming such that non diagonal elements in a is zero
        for i in range(n):
            factor = a[i][k]
            if a[i][k] == 0 or i==k:
                continue
            for j in range(k, n):
                a[i][j] = a[i][j] - factor * a[k][j]
            
            b[i] = b[i] - factor * b[k]
    return b, a