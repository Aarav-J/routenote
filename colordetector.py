import cv2, numpy as np 

BANDS = { 
    "red": [(0,9), (170,179)], 
    "orange": [(10,25)],
    "yellow": [(26,34)],
    "green": [(35,85)],
    "cyan": [(86, 97)], 
    "blue": [(98,125)],
    "purple": [(126,145)],
    "pink": [(146,169)]
}

def classify_color(bgr_crop: np.ndarray):
    """
    Simple but effective color extractor using HSV dominant color
    Returns the color name and the dominant HSV value
    """
    # Simple preprocessing with Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(bgr_crop, (5, 5), 0)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create a mask for non-white, non-black, and non-gray pixels
    h, s, v = cv2.split(hsv)
    
    # Use K-means to find the dominant color (more robust than mean/median)
    pixels = blurred.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    # Only use pixels with reasonable saturation for clustering
    mask = (s.flatten() > 20) & (v.flatten() > 30)
    if np.sum(mask) > 10:  # If we have enough colorful pixels
        pixels = pixels[mask]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 1  # Extract only the most dominant color
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Convert dominant color to HSV
    dominant_color = centers[0].astype(np.uint8).reshape(1, 1, 3)
    dominant_hsv = cv2.cvtColor(dominant_color, cv2.COLOR_BGR2HSV)[0][0]
    
    # Get color name for reference
    h, s, v = dominant_hsv
    
    color_name = "unknown"
    if s < 30:
        if v > 180:
            color_name = "white"
        elif v < 50:
            color_name = "black"
        else:
            color_name = "gray"
    else:
        # Use the BANDS to classify
        hue = h
        for name, ranges in BANDS.items():
            for low, high in ranges:
                if low <= hue <= high:
                    color_name = name
                    break
    
    # Return color name and HSV tuple
    return color_name, (int(h), int(s), int(v))

def enhanced_classify_color(bgr_crop): 
    mean_val = cv2.mean(bgr_crop)[0]
    gamma = 1.2 if mean_val < 100 else (0.8 if mean_val > 150 else 1.0)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256): 
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    adjusted = cv2.LUT(bgr_crop, lookUpTable)
    blurred = cv2.bilateralFilter(adjusted, 9, 75, 75)
    pixels = blurred.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    
    k = 3
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    hsv_centers = []
    for center in centers: 
        color = center.astype(np.uint8).reshape(1,1,3)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
        hsv_centers.append((hsv, (hsv[1] * hsv[2]) / 255.0 ))

        hsv_centers.sort(key=lambda x: x[1], reverse=True)
        for hsv, _ in hsv_centers:
            if 20 < hsv[2] < 230: 
                return "color", (int(hsv[0]), int(hsv[1]), int(hsv[2]))
        return "color", (int(hsv_centers[0][0][0]), int(hsv_centers[0][0][1]), int(hsv_centers[0][0][2]))
