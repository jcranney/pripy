import numpy as np

gap_width_m = 0.359*0
seg_width_m = 8.4

def make_pupil(x0,y0,x_res,y_res,seg_diam):
    return make_phimat(x0,y0,x_res,y_res,seg_diam).sum(axis=0)

def make_phimat(x0,y0,x_res,y_res,seg_diam):
    yy,xx = np.mgrid[:y_res,:x_res]*1.0
    zz = np.tile(xx[None,:,:]*0,[7,1,1])
    zz[0,:,:] = (((xx-x0)**2+(yy-y0)**2)**0.5 < seg_diam/2)*1.0
    for i,theta in enumerate(np.linspace(0,2*np.pi,7)[:-1]):
        radial_offset = seg_diam*(1+gap_width_m/seg_width_m)
        zz[1+i,:,:] = (((xx-x0-radial_offset*np.cos(theta))**2+(yy-y0-radial_offset*np.sin(theta))**2)**0.5 < seg_diam/2)*1.0
    return zz

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()

    full_width = 600
    seg_diam = full_width/(3+gap_width_m/seg_width_m*2)

    plt.matshow(make_pupil(450,285,800,600,seg_diam))