from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=15, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    indexes = []
    uvs = []

    kernel_x = np.array([[1, 0, -1]])
    kernel_y = np.array([[1], [0], [-1]])

    # normalize images:
    im1 = im1 / 255
    im2 = im2 / 255

    fx = cv2.filter2D(im1, -1, kernel_x)
    fy = cv2.filter2D(im1, -1, kernel_y)
    ft = im1 - im2

    m = int(win_size / 2)  # window_size is odd, all the pixels with offset in between [-w, w] are inside

    for i in range(m + step_size, im1.shape[0] - m, step_size):
        for j in range(m + step_size, im1.shape[1] - m, step_size):
            # for windows for the pixel (i,j):
            Ix = fx[i - m:i + m + 1, j - m:j + m + 1].flatten()
            Iy = fy[i - m:i + m + 1, j - m:j + m + 1].flatten()
            It = ft[i - m:i + m + 1, j - m:j + m + 1].flatten()

            A_1 = np.sum(np.matmul(Ix, Ix))
            A_2 = np.sum(np.matmul(Ix, Iy))
            A_3 = np.sum(np.matmul(Iy, Ix))
            A_4 = np.sum(np.matmul(Iy, Iy))

            D_1 = np.sum(np.matmul(Ix, It))
            D_2 = np.sum(np.matmul(Iy, It))

            mat = np.linalg.pinv(np.array([[A_1, A_2], [A_3, A_4]]))
            B = np.array([[-D_1], [-D_2]])
            u_v = np.matmul(mat, B)

            uvs.append([u_v[0][0], u_v[1][0]])
            indexes.append([j, i])

    return np.array(indexes), np.array(uvs)


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    rows, cols = img.shape[0], img.shape[1]
    rows = rows % pow(2, levels)
    cols = cols % pow(2, levels)
    if rows != 0:
        img = img[:-rows, :]
    if cols != 0:
        img = img[:, :-cols]

    pyr_list = [img]
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel_gau = cv2.getGaussianKernel(k_size, sigma)

    for i in range(1, levels):
        img_temp = cv2.filter2D(pyr_list[i - 1], -1, kernel=kernel_gau)
        img_temp = img_temp[::2, ::2]  # increase the size of image
        pyr_list.append(img_temp)
    return pyr_list


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    gs_k = 4 * gs_k

    if isGray(img):
        # zero padding:
        expand_img = np.zeros((2 * img.shape[0], 2 * img.shape[1]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                expand_img[2 * i][2 * j] = img[i][j]
        # blur the image with gaussian kernel:
        expand_img = cv2.filter2D(expand_img, -1, kernel=gs_k)

    else:
        # zero padding:
        expand_img = np.zeros((2 * img.shape[0], 2 * img.shape[1], 3))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(3):
                    expand_img[2 * i][2 * j][k] = img[i][j][k]
        # blur the image with gaussian kernel:
        expand_img = cv2.filter2D(expand_img, -1, kernel=gs_k)

    return expand_img


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gau_pyr = gaussianPyr(img, levels)
    lap_pyr = [gau_pyr[levels - 1]]

    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel = kernel.dot(kernel.T)

    for i in range(levels - 1, 0, -1):
        org_img = gau_pyr[i - 1]
        expand_img = gaussExpand(gau_pyr[i], kernel)
        differance = org_img - expand_img
        lap_pyr.append(differance)

    lap_pyr.reverse()
    return lap_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel = kernel.dot(kernel.T)

    restore_img = lap_pyr[levels - 1]
    for i in range(levels - 2, -1, -1):
        restore_img = lap_pyr[i] + gaussExpand(restore_img, kernel)

    return restore_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    is_gray = isGray(img_1)

    # blending without using the pyramid.
    simple_blend = simpleBlend(img_1, img_2, mask, is_gray)

    # blending with using the pyramid.
    lap_pyr1 = laplaceianReduce(img_1, levels)
    lap_pyr2 = laplaceianReduce(img_2, levels)
    gaus_maks = gaussianPyr(mask, levels)

    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel = kernel.dot(kernel.T)

    prev_img = simpleBlend(lap_pyr1[levels - 1], lap_pyr2[levels - 1], gaus_maks[levels - 1], is_gray)
    for i in range(levels - 2, -1, -1):
        new_img = simpleBlend(lap_pyr1[i], lap_pyr2[i], gaus_maks[i], is_gray)
        expand_img = gaussExpand(prev_img, kernel)
        prev_img = new_img + expand_img

    simple_blend = simple_blend[0:prev_img.shape[0], 0:prev_img.shape[1]]
    # print("simple_blend:", simple_blend.shape)
    # print("pyr_blend:", prev_img.shape)
    return simple_blend, prev_img


def simpleBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, is_gray: bool) -> np.ndarray:
    simple_blend = np.zeros_like(img_1)
    if is_gray:
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x][y] == 1:
                    simple_blend[x][y] = img_1[x][y]
                else:
                    simple_blend[x][y] = img_2[x][y]
    else:
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x][y][1] == 1:
                    simple_blend[x][y] = img_1[x][y]
                else:
                    simple_blend[x][y] = img_2[x][y]
    return simple_blend


def isGray(img: np.ndarray) -> bool:
    ans = True
    if len(img.shape) is 3:
        ans = False
    return ans
