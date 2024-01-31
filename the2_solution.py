# Andaç Berkay Seval 2235521
# Asrın Doğrusöz 2380301

import cv2
import os
import numpy as np
import scipy.fftpack as fp 
from skimage import io, img_as_ubyte
from scipy.linalg import hadamard
from skimage import exposure

INPUT_PATH = "./THE2_images/"
OUTPUT_PATH = "./Outputs/"

def read_image(img_path, rgb = True):
    img = io.imread(img_path)
    return img

def write_image(img, output_path, rgb = True):
    io.imsave(output_path, img)

def fourier_transformation(img):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = abs(r)
    g = abs(g)
    b = abs(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    ### for normalizaion
    # if n == 1:
    #     outr = np.zeros((814, 1600))
    #     outg = np.zeros((814, 1600))
    #     outb = np.zeros((814, 1600))
    #     outr = cv2.normalize(r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outg = cv2.normalize(g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outb = cv2.normalize(b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     return cv2.merge((outr, outg, outb))
    # else:
    #     outr = np.zeros((3168, 4752))
    #     outg = np.zeros((3168, 4752))
    #     outb = np.zeros((3168, 4752))
    #     outr = cv2.normalize(r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outg = cv2.normalize(g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outb = cv2.normalize(b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     return cv2.merge((outr, outg, outb))
    return r + g + b

def discrete_cosine_transformation(img):
    r, g, b = cv2.split(img)
    r = fp.dctn(r)
    g = fp.dctn(g)
    b = fp.dctn(b)
    ### for normalizaion
    # if n == 1:
    #     outr = np.zeros((814, 1600))
    #     outg = np.zeros((814, 1600))
    #     outb = np.zeros((814, 1600))
    #     outr = cv2.normalize(r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outg = cv2.normalize(g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outb = cv2.normalize(b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     return cv2.merge((outr, outg, outb))
    # else:
    #     outr = np.zeros((3168, 4752))
    #     outg = np.zeros((3168, 4752))
    #     outb = np.zeros((3168, 4752))
    #     outr = cv2.normalize(r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outg = cv2.normalize(g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     outb = cv2.normalize(b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     return cv2.merge((outr, outg, outb))            
    return r + g + b

def hadamard_transformation(img, n):
    black = [0, 0, 0]
    if n == 1:
        img2 = cv2.copyMakeBorder(img,0,2048-814,0,2048-1600,cv2.BORDER_CONSTANT,value=black)
        r, g, b = cv2.split(img2)
        h = hadamard(2048)
        r = np.matmul(h, np.matmul(r, h.transpose()))
        g = np.matmul(h, np.matmul(g, h.transpose()))
        b = np.matmul(h, np.matmul(b, h.transpose()))
        ### for normalizaion
        # outr = np.zeros((2048, 2048))
        # outg = np.zeros((2048, 2048))
        # outb = np.zeros((2048, 2048))
        # outr = cv2.normalize(r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # outg = cv2.normalize(g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # outb = cv2.normalize(b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # return cv2.merge((outr, outg, outb))
        return r + g + b
    else:
        img2 = cv2.copyMakeBorder(img,0,8192-3168,0,8192-4752,cv2.BORDER_CONSTANT,value=black)
        r, g, b = cv2.split(img2)
        h = hadamard(8192)
        r = np.matmul(h, np.matmul(r, h.transpose()))
        g = np.matmul(h, np.matmul(g, h.transpose()))
        b = np.matmul(h, np.matmul(b, h.transpose()))
        ### for normalizaion
        # outr = np.zeros((8192, 8192))
        # outg = np.zeros((8192, 8192))
        # outb = np.zeros((8192, 8192))
        # outr = cv2.normalize(r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # outg = cv2.normalize(g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # outb = cv2.normalize(b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # return cv2.merge((outr, outg, outb))
        return r + g + b
    ### Due to the long runtime, an alternative way is implemented. Instead of padding with black values, 
    ### image is resized with closest 2^2j dimensions. However, for image 2, it has still a long runtime.
    ### Hence, program can be frozen in runtime.
    # if n == 1:
    #     img2 = cv2.resize(img, (1024, 1024))
    #     r, g, b = cv2.split(img2)
    #     h = hadamard(1024)
    #     r = np.matmul(h, np.matmul(r, h.transpose()))
    #     g = np.matmul(h, np.matmul(g, h.transpose()))
    #     b = np.matmul(h, np.matmul(b, h.transpose()))
    #     return r + g + b
    # else:
    #     img2 = cv2.resize(img, (4096, 4096))
    #     r, g, b = cv2.split(img2)
    #     h = hadamard(4096)
    #     r = np.matmul(h, np.matmul(r, h.transpose()))
    #     g = np.matmul(h, np.matmul(g, h.transpose()))
    #     b = np.matmul(h, np.matmul(b, h.transpose()))
    #     return r + g + b

def ideal_low_pass_filter(img, cf):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    h = np.zeros((960,1280))
    for i in range(960):
        for j in range(1280):
            distance = np.sqrt((i - 480)**2 + (j - 640)**2)
            if distance <= cf:
                h[i, j] = 1
    lpfr = r * h
    lpfg = g * h
    lpfb = b * h
    lpfr = fp.ifftshift(lpfr)
    lpfg = fp.ifftshift(lpfg)
    lpfb = fp.ifftshift(lpfb)
    output_r = np.real(fp.ifft2(lpfr))
    output_g = np.real(fp.ifft2(lpfg))
    output_b = np.real(fp.ifft2(lpfb))

    outr = np.zeros((960, 1280))
    outg = np.zeros((960, 1280))
    outb = np.zeros((960, 1280))

    outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge((outr, outg, outb)), h

def ideal_high_pass_filter(img, cf):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    h = np.ones((960,1280))
    for i in range(960):
        for j in range(1280):
            distance = np.sqrt((i - 480)**2 + (j - 640)**2)
            if distance <= cf:
                h[i, j] = 0
    lpfr = r * h
    lpfg = g * h
    lpfb = b * h
    lpfr = fp.ifftshift(lpfr)
    lpfg = fp.ifftshift(lpfg)
    lpfb = fp.ifftshift(lpfb)
    output_r = np.real(fp.ifft2(lpfr))
    output_g = np.real(fp.ifft2(lpfg))
    output_b = np.real(fp.ifft2(lpfb))

    outr = np.zeros((960, 1280))
    outg = np.zeros((960, 1280))
    outb = np.zeros((960, 1280))

    outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge((outr, outg, outb)), h

def gaussian_low_pass_filter(img, cf):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    h = np.zeros((960,1280))
    for i in range(960):
        for j in range(1280):
            distance_2 = (i - 480)**2 + (j - 640)**2
            ### Different gaussian filters are implemented. However, the last one is the most suitable.
            # h[i, j] = np.exp(distance_2 / (2 * cf)) / (2 * np.pi * cf)
            # h[i, j] = np.exp(-2 * (np.pi)**2 * distance_2 * cf**2)
            # h[i, j] = np.exp(-distance_2 / (2 * cf**2)) / (2 * np.pi * cf**2)
            h[i, j] = np.exp(-distance_2 / (2 * cf**2))
    lpfr = r * h
    lpfg = g * h
    lpfb = b * h
    lpfr = fp.ifftshift(lpfr)
    lpfg = fp.ifftshift(lpfg)
    lpfb = fp.ifftshift(lpfb)
    output_r = np.real(fp.ifft2(lpfr))
    output_g = np.real(fp.ifft2(lpfg))
    output_b = np.real(fp.ifft2(lpfb))

    outr = np.zeros((960, 1280))
    outg = np.zeros((960, 1280))
    outb = np.zeros((960, 1280))

    outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge((outr, outg, outb)), h

def gaussian_high_pass_filter(img, cf):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    h = np.zeros((960,1280))
    for i in range(960):
        for j in range(1280):
            distance_2 = (i - 480)**2 + (j - 640)**2
            ### Different gaussian filters are implemented. However, the last one is the most suitable.
            # h[i, j] = 1 - np.exp(distance_2 / (2 * cf)) / (2 * np.pi * cf)
            # h[i, j] = 1 - np.exp(-(distance_2)**2 / (2 * cf**2)) / (2 * np.pi * cf**2)
            h[i, j] = 1 - np.exp(-distance_2 / (2 * cf**2))
    lpfr = r * h
    lpfg = g * h
    lpfb = b * h
    lpfr = fp.ifftshift(lpfr)
    lpfg = fp.ifftshift(lpfg)
    lpfb = fp.ifftshift(lpfb)
    output_r = np.real(fp.ifft2(lpfr))
    output_g = np.real(fp.ifft2(lpfg))
    output_b = np.real(fp.ifft2(lpfb))

    outr = np.zeros((960, 1280))
    outg = np.zeros((960, 1280))
    outb = np.zeros((960, 1280))

    outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge((outr, outg, outb)), h  
    # return cv2.merge((output_r, output_g, output_b)), hi    

def butterworth_low_pass_filter(img, cf):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    h = np.zeros((960,1280))
    for i in range(960):
        for j in range(1280):
            distance_2 = (i - 480)**2 + (j - 640)**2
            ### Different butterworth filters are implemented. However, the last one is the most suitable.
            ### n = 1 is taken and it is assumed to be second order due to 2n = 2.
            # h[i, j] = 1 / (1 + np.sqrt((distance_2)**4 / cf))
            h[i, j] = 1 / (1 + distance_2 / cf)
    lpfr = r * h
    lpfg = g * h
    lpfb = b * h
    lpfr = fp.ifftshift(lpfr)
    lpfg = fp.ifftshift(lpfg)
    lpfb = fp.ifftshift(lpfb)
    output_r = np.real(fp.ifft2(lpfr))
    output_g = np.real(fp.ifft2(lpfg))
    output_b = np.real(fp.ifft2(lpfb))

    outr = np.zeros((960, 1280))
    outg = np.zeros((960, 1280))
    outb = np.zeros((960, 1280))

    outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge((outr, outg, outb)), h 

def butterworth_high_pass_filter(img, cf):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    h = np.zeros((960,1280))
    for i in range(960):
        for j in range(1280):
            distance_2 = (i - 480)**2 + (j - 640)**2
            ### Different butterworth filters are implemented. However, the last one is the most suitable.
            ### n = 1 is taken and it is assumed to be second order due to 2n = 2.
            # h[i, j] = 1 - 1 / (1 + np.sqrt((distance_2)**4 / cf))
            h[i, j] = 1 - 1 / (1 + distance_2 / cf)
    lpfr = r * h
    lpfg = g * h
    lpfb = b * h
    lpfr = fp.ifftshift(lpfr)
    lpfg = fp.ifftshift(lpfg)
    lpfb = fp.ifftshift(lpfb)
    output_r = np.real(fp.ifft2(lpfr))
    output_g = np.real(fp.ifft2(lpfg))
    output_b = np.real(fp.ifft2(lpfb))

    outr = np.zeros((960, 1280))
    outg = np.zeros((960, 1280))
    outb = np.zeros((960, 1280))

    outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge((outr, outg, outb)), h    

def band_reject_filter(img, cf1, cf2, n):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    if n == 4:
        h = np.zeros((2888,3024))
        for i in range(2888):
            for j in range(3024):
                distance = np.sqrt((i - 1444)**2 + (j - 1512)**2)
                if cf1 < distance < cf2:
                    h[i, j] = 1
        lpfr = r * h
        lpfg = g * h
        lpfb = b * h
        lpfr = fp.ifftshift(lpfr)
        lpfg = fp.ifftshift(lpfg)
        lpfb = fp.ifftshift(lpfb)
        output_r = np.real(fp.ifft2(lpfr))
        output_g = np.real(fp.ifft2(lpfg))
        output_b = np.real(fp.ifft2(lpfb))

        outr = np.zeros((2888, 3024))
        outg = np.zeros((2888, 3024))
        outb = np.zeros((2888, 3024))

        outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.merge((outr, outg, outb)), h
    else:
        h = np.zeros((1536,2048))
        for i in range(1536):
            for j in range(2048):
                distance = np.sqrt((i - 768)**2 + (j - 1024)**2)
                if cf1 < distance < cf2:
                    h[i, j] = 1
        lpfr = r * h
        lpfg = g * h
        lpfb = b * h
        lpfr = fp.ifftshift(lpfr)
        lpfg = fp.ifftshift(lpfg)
        lpfb = fp.ifftshift(lpfb)
        output_r = np.real(fp.ifft2(lpfr))
        output_g = np.real(fp.ifft2(lpfg))
        output_b = np.real(fp.ifft2(lpfb))

        outr = np.zeros((1536, 2048))
        outg = np.zeros((1536, 2048))
        outb = np.zeros((1536, 2048))

        outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.merge((outr, outg, outb)), h

def band_pass_filter(img, cf1, cf2, n):
    r, g, b = cv2.split(img)
    r = fp.fft2(r)
    g = fp.fft2(g)
    b = fp.fft2(b)
    r = fp.fftshift(r)
    g = fp.fftshift(g)
    b = fp.fftshift(b)
    if n == 4:
        h = np.ones((2888,3024))
        for i in range(2888):
            for j in range(3024):
                distance = np.sqrt((i - 1444)**2 + (j - 1512)**2)
                if cf1 < distance < cf2:
                    h[i, j] = 0
        lpfr = r * h
        lpfg = g * h
        lpfb = b * h
        lpfr = fp.ifftshift(lpfr)
        lpfg = fp.ifftshift(lpfg)
        lpfb = fp.ifftshift(lpfb)
        output_r = np.real(fp.ifft2(lpfr))
        output_g = np.real(fp.ifft2(lpfg))
        output_b = np.real(fp.ifft2(lpfb))

        outr = np.zeros((2888, 3024))
        outg = np.zeros((2888, 3024))
        outb = np.zeros((2888, 3024))

        outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.merge((outr, outg, outb)), h
    else:
        h = np.ones((1536,2048))
        for i in range(1536):
            for j in range(2048):
                distance = np.sqrt((i - 768)**2 + (j - 1024)**2)
                if cf1 < distance < cf2:
                    h[i, j] = 0
        lpfr = r * h
        lpfg = g * h
        lpfb = b * h
        lpfr = fp.ifftshift(lpfr)
        lpfg = fp.ifftshift(lpfg)
        lpfb = fp.ifftshift(lpfb)
        output_r = np.real(fp.ifft2(lpfr))
        output_g = np.real(fp.ifft2(lpfg))
        output_b = np.real(fp.ifft2(lpfb))

        outr = np.zeros((1536, 2048))
        outg = np.zeros((1536, 2048))
        outb = np.zeros((1536, 2048))

        outr = cv2.normalize(output_r, outr, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outg = cv2.normalize(output_g, outg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        outb = cv2.normalize(output_b, outb, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.merge((outr, outg, outb)), h    



if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    img = read_image(INPUT_PATH + "1.png")

    # fourier transformation for image 1
    out = fourier_transformation(img)
    write_image(out, OUTPUT_PATH + "F1.png")

    # hadamard transformation for image 1
    out = hadamard_transformation(img, 1)
    write_image(out, OUTPUT_PATH + "H1.png")

    # discrete cosine transformation for image 1
    out = discrete_cosine_transformation(img)
    write_image(out, OUTPUT_PATH + "C1.png")

    img = read_image(INPUT_PATH + "2.png")

    # fourier transformation for image 2
    out = fourier_transformation(img)
    write_image(out, OUTPUT_PATH + "F2.png")

    # hadamard transformation for image 2. It is commented since it can freeze the program. It has a very long runtime
    # due to creating a big hadamard matrix, taking transpose of it and matrix multiplications. It is recommended to run
    # this alone after the other ones are commented.
    # out = hadamard_transformation(img, 2)
    # write_image(out, OUTPUT_PATH + "H2.png")

    # discrete cosine transformation for image 2
    out = discrete_cosine_transformation(img)
    write_image(out, OUTPUT_PATH + "C2.png")

    img = read_image(INPUT_PATH + "3.png")

    # ideal low pass filter for image 3 with cut-off frequency 10
    out, h = ideal_low_pass_filter(img, 10)
    write_image(out, OUTPUT_PATH + "ILP_10.png")

    # ideal low pass filter for image 3 with cut-off frequency 50
    out, h = ideal_low_pass_filter(img, 50)
    write_image(out, OUTPUT_PATH + "ILP_50.png")

    # ideal low pass filter for image 3 with cut-off frequency 100
    out, h = ideal_low_pass_filter(img, 100)
    write_image(out, OUTPUT_PATH + "ILP_100.png")

    # gaussian low pass filter for image 3 with cut-off frequency 10
    out, h = gaussian_low_pass_filter(img, 10)
    write_image(out, OUTPUT_PATH + "GLP_10.png") 

    # gaussian low pass filter for image 3 with cut-off frequency 50
    out, h = gaussian_low_pass_filter(img, 50)
    write_image(out, OUTPUT_PATH + "GLP_50.png") 

    # gaussian low pass filter for image 3 with cut-off frequency 100
    out, h = gaussian_low_pass_filter(img, 100)
    write_image(out, OUTPUT_PATH + "GLP_100.png")

    # butterworth low pass filter for image 3 with cut-off frequency 10
    out, h = butterworth_low_pass_filter(img, 10)
    write_image(out, OUTPUT_PATH + "BLP_10.png")

    # butterworth low pass filter for image 3 with cut-off frequency 50
    out, h = butterworth_low_pass_filter(img, 50)
    write_image(out, OUTPUT_PATH + "BLP_50.png")

    # butterworth low pass filter for image 3 with cut-off frequency 100
    out, h = butterworth_low_pass_filter(img, 100)
    write_image(out, OUTPUT_PATH + "BLP_100.png")

    # ideal high pass filter for image 3 with cut-off frequency 10
    out, h = ideal_high_pass_filter(img, 10)
    write_image(out, OUTPUT_PATH + "IHP_10.png")

    # ideal high pass filter for image 3 with cut-off frequency 50
    out, h = ideal_high_pass_filter(img, 50)
    write_image(out, OUTPUT_PATH + "IHP_50.png")

    # ideal high pass filter for image 3 with cut-off frequency 100
    out, h = ideal_high_pass_filter(img, 100)
    write_image(out, OUTPUT_PATH + "IHP_100.png")

    # gaussian high pass filter for image 3 with cut-off frequency 10
    out, h = gaussian_high_pass_filter(img, 10)
    write_image(out, OUTPUT_PATH + "GHP_10.png") 

    # gaussian high pass filter for image 3 with cut-off frequency 50
    out, h = gaussian_high_pass_filter(img, 50)
    write_image(out, OUTPUT_PATH + "GHP_50.png") 

    # gaussian high pass filter for image 3 with cut-off frequency 100
    out, h = gaussian_high_pass_filter(img, 100)
    write_image(out, OUTPUT_PATH + "GHP_100.png") 

    # butterworth high pass filter for image 3 with cut-off frequency 10
    out, h = butterworth_high_pass_filter(img, 10)
    write_image(out, OUTPUT_PATH + "BHP_10.png")

    # butterworth high pass filter for image 3 with cut-off frequency 50
    out, h = butterworth_high_pass_filter(img, 50)
    write_image(out, OUTPUT_PATH + "BHP_50.png")

    # butterworth high pass filter for image 3 with cut-off frequency 100
    out, h = butterworth_high_pass_filter(img, 100)
    write_image(out, OUTPUT_PATH + "BHP_100.png")

    img = read_image(INPUT_PATH + "4.png")

    # band reject filter for image 4 with cut-off frequencies 2 and 50
    out, h = band_reject_filter(img, 2, 50, 4)
    write_image(out, OUTPUT_PATH + "BR1.png")

    # band pass filter for image 4 with cut-off frequencies 2 and 50
    out, h = band_pass_filter(img, 2, 50, 4)
    write_image(out, OUTPUT_PATH + "BP1.png")

    img = read_image(INPUT_PATH + "5.png")

    # band reject filtler for image 5 with cut-off frequencies 2 and 85
    out, h = band_reject_filter(img, 2, 85, 5)
    write_image(out, OUTPUT_PATH + "BR2.png")

    # band pass filtler for image 5 with cut-off frequencies 2 and 85
    out, h = band_pass_filter(img, 2, 85, 5)
    write_image(out, OUTPUT_PATH + "BP2.png")

    # adaptive histogram equalization for improving contrast of image 6
    img = read_image(INPUT_PATH + "6.png")
    out = exposure.equalize_adapthist(img)
    out = img_as_ubyte(out)
    write_image(out, OUTPUT_PATH + "Space6.png")

    # adaptive histogram equalization for improving contrast of image 7
    img = read_image(INPUT_PATH + "7.png")
    out = exposure.equalize_adapthist(img)
    out = img_as_ubyte(out)
    write_image(out, OUTPUT_PATH + "Space7.png")