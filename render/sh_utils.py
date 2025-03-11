
import numpy as np

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def rgb_from_sh(deg,sh,xyz = None,camera_position = None):
    
    if deg >= 0:
        result = (C0 * sh[:,0:3])

    
    if deg >= 1:
        direction = (xyz - camera_position.T)/np.linalg.norm([xyz - camera_position.T],axis = 1)
        x = direction[:,0][np.newaxis,:].T
        y = direction[:,1][np.newaxis,:].T
        z = direction[:,2][np.newaxis,:].T
        result =  result - C1 * y * sh[:,[3,4,5]] + C1 * z * sh[:,[6,7,8]] - C1 * x* sh[:,[9,10,11]]

    if deg >= 2:
        xx,yy,zz = x * x,y * y,z * z
        xy,yz,xz = x * y, y * z, x * z
        result = result + C2[0] * xy * sh[:,[12,13,14]]
        + C2[1] * yz * sh[:,[15,16,17]]
        + C2[2] * (2.0 * zz - xx - yy) * sh[:,[18,19,20]] 
        + C2[3] * xz * sh[:,[21,22,23]] + C2[4] * (xx - yy) * sh[:,[24,25,26]]

    if deg > 2:
        result = result + C3[0] * y * (3.0 * xx - yy) * sh[:,[27,28,29]] +C3[1] * xy * z * sh[:,[30,31,32]] 
        +C3[2] * y * (4.0 * zz - xx - yy) * sh[:,[33,34,35]] +C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)*sh[:,[36,37,38]] 
        +C3[4] * x * (4.0 * zz - xx - yy) * sh[:,[39,40,41]] +C3[5] * z * (xx - yy) * sh[:,[42,43,44]]
        +C3[6] * x * (xx - 3.0 * yy) * sh[:,[45,46,47]]

    # Final adjustment
    result += 0.5
    return result

