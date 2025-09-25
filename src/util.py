import cv2
import numpy as np

def filter_mask(closing_kernel_size, opening_kernel_size, mask):
    """
    Given a mask, apply morphological filters (closing followed by opening) 
    to filter out unwanted pixels.

    Parameters:
    -----------
       closing_kernel_size : Int
                             Size of the closing kernel in pixels.
       opening_kernel_size : Int
                             Size of the opening kernel in pixels.
        mask : Array[int]
               A binary mask.
    
    Returns:
    --------
        mask_closed_opened : Array[int]
               A morphologically filtered binary mask.
    """

    ## Closing filter: Remove empty pixels within mask
    # Create a kernel element which is closing_kernel_size^2 in size
    closing_kernel_element = (closing_kernel_size, closing_kernel_size)
    # Create a closing filter kernel
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               closing_kernel_element)
    # Apply closing filter to input mask
    mask_closed = cv2.morphologyEx(np.nan_to_num(mask), cv2.MORPH_CLOSE,
                                   closing_kernel)

    ## Opening filter: Removing filled pixels outside of mask
    # Create a kernel element which is closing_kernel_size^2 in size
    opening_kernel_element = (opening_kernel_size, opening_kernel_size)
    # Create an opening filter kernel
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               opening_kernel_element)
    # Apply opening filter to closed mask
    mask_closed_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,
                                          opening_kernel)

    # Ensure the clipped areas remain clipped
    mask_closed_opened[mask_closed_opened == 0] = np.nan

    return mask_closed_opened