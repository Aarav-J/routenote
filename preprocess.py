import cv2, numpy as np

def gray_world_white_balance(bgr):

    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0 + 1e-6
    b *= m / mb; g *= m / mg; r *= m / mr
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def auto_gamma(bgr, target_mean=140):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cur = gray.mean() + 1e-6
    gamma = np.clip(np.log(target_mean/255.0) / np.log(cur/255.0), 0.5, 1.5)
    lut = ((np.arange(256)/255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)

def clahe_on_v(bgr, clip=2.0, tile=(8,8)):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    v = clahe.apply(v)
    hsv = cv2.merge([h,s,v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def light_denoise_sharpen(bgr):
    den = cv2.bilateralFilter(bgr, d=5, sigmaColor=35, sigmaSpace=35)
    blur = cv2.GaussianBlur(den, (0,0), 1.0)
    sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
    return sharp
def preprocess_wall(bgr): 
    out = gray_world_white_balance(bgr)
    out = auto_gamma(out)
    out = clahe_on_v(out)
    out = light_denoise_sharpen(out)
    return out
if __name__ == "__main__":
    img = cv2.imread("./final_images/009.jpg")
    out = preprocess_wall(img)
    cv2.imwrite("testout.jpg", out)