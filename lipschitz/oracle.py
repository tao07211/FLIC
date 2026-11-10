import numpy as np

def estimate_lipschitz_rectangular(image_array, x_range=(-1, 1), y_range=(-1, 1), percentile=99.9):

    I = image_array.astype(np.float64)
    
    H, W = I.shape

    grad_y_pixel, grad_x_pixel = np.gradient(I, edge_order=2)             

    x_min, x_max = x_range
    y_min, y_max = y_range

    dx_coord = (x_max - x_min) / W                                      
    dy_coord = (y_max - y_min) / H                                      

    grad_x_scaled = grad_x_pixel / dx_coord
    grad_y_scaled = grad_y_pixel / dy_coord

    gradient_magnitude = np.sqrt(grad_x_scaled**2 + grad_y_scaled**2)

    lipschitz_constant = np.percentile(gradient_magnitude, percentile)

    return lipschitz_constant, grad_x_scaled, grad_y_scaled, gradient_magnitude