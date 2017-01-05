import numpy as np
from PIL import Image
import pdb
def rgb2Lab(RGB):
    """
    RGB2Lab takes matrices corresponding to Red, Green, and Blue, and 
    transforms them into CIELab.  
    """
    RGB = np.array(RGB)   
    if (np.max(RGB)>1.0):
        RGB=RGB/255.0
    R = RGB[:,:,0]
    G = RGB[:,:,1]
    B = RGB[:,:,2]
    
    s=R.size
    (M,N) = R.shape
    # Set a threshold
    T = 0.008856;
    R=R.reshape((1,s))
    G=G.reshape((1,s))
    B=B.reshape((1,s))
    RGB=np.vstack((R,G,B))
    
   
    # RGB to XYZ
    MAT = np.array([[0.412453, 0.357580, 0.180423],
       [0.212671, 0.715160, 0.072169],
       [0.019334, 0.119193, 0.950227]])
    XYZ = MAT.dot(RGB)
    X = XYZ[0,:] / 0.950456
    Y = XYZ[1,:]
    Z = XYZ[2,:] / 1.088754
    XT = X > T
    YT = Y > T
    ZT = Z > T
    fX = XT * X**(1.0/3) + (~XT) * (7.787 * X + 16.0/116)

    #Compute L
    Y3 = Y**(1.0/3); 
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16.0/116)
    L  = YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y)

    fZ = ZT * Z**(1/3) + (~ZT) * (7.787 * Z + 16.0/116)
    # Compute a and b
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)
    
    L = np.reshape(L, (M, N))
    a = np.reshape(a, (M, N))
    b = np.reshape(b, (M, N))
    
    return (L,a,b)

if __name__ == '__main__':
    img = Image.open("sun_aaegrbxogokacwmz.jpg")
    Lab= rgb2Lab(img)
    print Lab