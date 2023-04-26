# # This is a sample Python script
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
from copy import copy
import openpyxl
from scipy.optimize import curve_fit
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Tuple, List
import itk


#https://github.com/yaelsuarez/uuutils/tree/main/MRI
#批量获取文件名称以及路径
def get_files(file_path, extension):
    #for each
    filename = [str(f) for f in os.listdir(file_path) if f.endswith(extension)]
    file_paths = [os.path.join(file_path, f) for f in filename]
    return file_paths, filename

#获取灰色图像数据
def get_gray(img):
    #数据类型转换，转型为32位浮点类型
    img = img.astype(np.float32)
    #对数组中每个数据进行处理，转成unsignInt(8字节) unsignedInteger
    return (img*255/np.max(img)).astype(np.uint8)

#通过dicom读取数据
def read_dicom(file):
    #解析读取图像文件，获取pydicom.dataset.FileDataset
    ds = pydicom.dcmread(file)
    #图像数据元组
    image = ds.pixel_array
    gray = get_gray(image)
    return image, gray

def read_fdf(filepath):
    imageio = itk.FDFImageIO.New()
    print(filepath)
    image = itk.GetArrayFromImage(itk.imread(filepath, imageio=imageio))
    gray = get_gray(image)
    return image, gray

#opencv 通过灰色化数据，检测图片中的圆形斑点 reference:https://blog.csdn.net/m0_63235356/article/details/124157510
def detect_circles(image_grayscale):
    # Create a SimpleBlobDetector object
    params = cv2.SimpleBlobDetector_Params()
    # Set the threshold and filter by color
    params.minThreshold = 5
    params.maxThreshold = 300
    params.filterByColor = True
    params.blobColor = 255
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs in the image
    keypoints = detector.detect(image_grayscale)
    # Draw detected blobs as circles on the image
    # im_with_keypoints = cv2.drawKeypoints(image_grayscale, keypoints, np.array([]), (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints #, im_with_keypoints

def get_masks(keypoints, image):
    masks = []
    for idx, keypoint in enumerate(keypoints):
        # mask初始化：[[0,0,0...0],[0,0,0...0]...[0,0,0...0]]
        mask = np.zeros_like(image, dtype=np.uint8)
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        r = int(keypoint.size*0.95/2) #偏差值（0.05偏差） 计算斑点半径
        mask = cv2.circle(mask, [x, y], r, (255, 255, 255), -1).astype(np.bool_) #在一个和原图片（image）大小一致的图片中（mask）画一个半径为r，像素坐标点为（x，y）的圆，其中数据表示为标识上的像素点为true，否则为false
        masks.append({'ind':idx, "x": x, "y": y, "r": r, "mask": mask})
    return masks

#求圆形平均亮度
def get_mean_intensities(image, masks, filename):
    means = []
    for m in masks:
        m = copy(m)
        #np.mean 取平均值
        #把mask对应image中的坐标的值取出来到一个数组中，取平均值
        test = image[m['mask']]
        mean_intensity = np.mean(test)
        m['mean_intensity'] = mean_intensity
        m['filename'] = filename
        means.append(m)
    return means

def plot_proof(ax, dict, image, name):
    keys = dict[0].keys()
    values = [[i[k] for k in keys] for i in dict]
    for x, y, r, mean in values:
        ax.text(x, y, str(mean), fontsize='small', color='white')
    ax.imshow(image)
    ax.set_title(name[:-4])

def show_masks(image, masks, fn):
    mask_arrays = np.array([m['mask'] for m in masks])
    print(mask_arrays.shape)
    alpha = np.sum(mask_arrays, axis=0).astype(np.float32)*0.5
    R = np.ones_like(alpha, dtype=np.float32)*alpha
    G = np.zeros_like(alpha, dtype=np.float32)*alpha
    B = R
    all_masks = np.stack([R,G,B,alpha], -1)
    plt.imshow(image, cmap='gray')
    plt.imshow(all_masks)
    for m in masks:
        plt.text(m['x'], m['y'], m['ind'])
    plt.title(fn)
    plt.show()

@dataclass
class ManualKeypoint:
    #point(x,y)
    pt: Tuple[float, float]
    size: float

def get_masks_from_basefile(first_file, manual_keypoints:List[ManualKeypoint]=None):
    mask_origin = first_file
    # 读取dicom图片，灰色
    image, image_grayscale = read_dicom(mask_origin)
    # 读取图片中的圆形斑点
    keypoints = detect_circles(image_grayscale)

    if manual_keypoints == None: manual_keypoints = list()
    #将人工识别点添加到数据中
    keypoints = manual_keypoints + [kp for kp in keypoints]
    masks = get_masks(keypoints, image)
    return masks

#实验样本亮度
def get_samples_intensities(file_paths, filename, masks, yn):
    samples = []
    # f:file fn:filename
    for f, fn in tqdm(zip(file_paths, filename)):
        image, image_grayscale = read_dicom(f)
        if yn == "Yes":
            show_masks(image, masks, fn)
        samples += get_mean_intensities(image, masks, fn)
    return samples

#读取csv文件到ManualKeypoint列表
def read_kp_csv(csv_path):
    if csv_path==None: return None
    df = pd.read_csv(csv_path)
    records = df.to_dict('records')
    # to_dict('records'), 一行行读取数据（x，y，size）
    return [ManualKeypoint((r['x'], r['y']), r['size']) for r in records]


def graph_echotime_vs_intensity_disordered(df_mri, echo_time):
    legend = df_mri["Concentration"].unique()
    all_rs = []
    concentrations = []
    for i, l in zip(df_mri.ind.unique(), legend):
        y = df_mri[df_mri.ind == i].mean_intensity.values
        x = echo_time
        popt = fit_r(x, y)
        I0 = popt[0]
        rs = popt[1]
        all_rs.append(rs)
        concentrations.append(l)
        plt.scatter(x, y, label=l)
        plt.plot(x, I0 * np.exp(-rs * x))
    all_r = np.array(all_rs)
    plt.legend(title="Fe concentration (mM)", title_fontsize=font_size, bbox_to_anchor=(1, 1), loc='upper left',
               fontsize=font_size)
    plt.xlabel("Echo time (ms)", fontsize=font_size + 2)
    plt.ylabel("Intensity", fontsize=font_size + 2)
    plt.show()
    return all_r, concentrations


def graph_echotime_vs_intensity_ordered(df_mri, echo_time):
    all_rs = []
    concentrations = []
    for i, l in zip(sorted(df_mri.ind.unique(), key=lambda x: df_mri[df_mri.ind == x].Concentration.unique()[0]),
                    sorted(df_mri.Concentration.unique())):
        y = df_mri[df_mri.ind == i].mean_intensity.values
        x = echo_time
        if parameter != "T1":
            if l != -1:
                popt = fit_r(x, y)
                I0 = popt[0]
                rs = popt[1]
                all_rs.append(rs)
                concentrations.append(l)
                plt.scatter(x, y, label=l)
                plt.plot(x, I0 * np.exp(-rs * x))
        if "T1" in parameter:
            if l != -1:
                y_correct = []
                min = np.argmin(y)
                for i, v in enumerate(y):
                    if i <= min:
                        y_correct.append(v * -1)
                    else:
                        y_correct.append(v)
                popt = fit_r_inv(x, y_correct)
                A = popt[0]
                rs = popt[1]
                B = popt[2]
                all_rs.append(rs)
                concentrations.append(l)
                plt.scatter(x, y_correct, label=l)
                plt.plot(x, A * (np.exp(-x * rs)) + B)
    all_r = np.array(all_rs)
    plt.legend(title="Fe concentration (mM)", title_fontsize=font_size, bbox_to_anchor=(1, 1), loc='upper left',
               fontsize=font_size)
    plt.xlabel("Echo time (ms)", fontsize=font_size + 2)
    plt.ylabel("Intensity (a.u.)", fontsize=font_size + 2)
    plt.show()
    return all_r, concentrations


def fit_r(echo_time, mean_intensity):
    def func(x, I0, rs):
        return I0 * np.exp(-(rs) * x)

    popt, pcov = curve_fit(func, echo_time, mean_intensity, maxfev=20000)
    return popt


def fit_r_inv(echo_time, mean_intensity):
    def func(x, A, rs, B):
        return A * (np.exp(-x * rs)) + B

    popt, pcov = curve_fit(func, echo_time, mean_intensity, p0=[-10, 1, 1], maxfev=20000)
    return popt


def graph_r_vs_concentration(concentration, r):
    x = np.array(concentration[:-2])
    y = r[:-2]
    y_label = np.max(y) * 0.66
    slope, intercept, rvalue, pvalue, stderr = linregress(x, y)
    relaxivity = slope
    print("Relaxivity = ", relaxivity, "\nRsquared = ", rvalue * rvalue)
    r_squared = rvalue * rvalue
    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept, '-r')
    plt.xlabel("Fe concentration (mM)", fontsize=font_size + 2)
    plt.ylabel(label_R + "(s$^{-1}$)", fontsize=font_size + 2)
    plt.title("", fontsize=font_size)
    plt.show()
    plt.annotate("Relaxivity = {:.3f} mM$^-$$^1$s$^-$$^1$".format(slope), (np.min(x), y_label), fontsize=font_size)
    plt.annotate("R$^2$ = {:.4f}".format(r_squared), (np.min(x), y_label * 0.6), fontsize=font_size)


if __name__ == '__main__':
    # 初始化需要使用到的参数
    # 实验数据文件夹
    experiment_dir = r"/Users/tongchen/Downloads/Fe2O3 SiO2 0.1-2mM"
    file_type = "DICOM"
    parameter = "T1"  # T1, T2, or T2star
    # masks数据是否需要打印图像，Yes为需要
    print_masks = "No"  # "Yes" for printing mask circle selection, or "No" for not printing them
    file_dir = os.path.join(experiment_dir, file_type, parameter)
    # 对照组数据存放位置
    first_file = os.path.join(file_dir, "slice001image001echo001.dcm")
    # 实验信息
    info_csv_fp = os.path.join(experiment_dir, "info_mri.xlsx")
    font_size = 12
    # (For T2* no manual keypoints needed, for T2 and T2, activate manual keypoints)
    if "T1" in parameter:
        manual_keypoints_fp = os.path.join(experiment_dir, "manual_kp.csv")
        label_R = r"$\dfrac{1}{T1} $"
        # % % writefile
        # "{manual_keypoints_fp}"
        # x, y, size
        # 97, 51, 15
    if "T2" in parameter and "star" not in parameter:
        manual_keypoints_fp = os.path.join(experiment_dir, "manual_kp.csv")
        label_R = r"$\dfrac{1}{T2}$ "
        # % % writefile
        # "{manual_keypoints_fp}"
        # x, y, size
        # 95, 50, 15
    if "T2star" in parameter:
        manual_keypoints_fp = None
        label_R = r"$\dfrac{1}{T2*}$ "

    # Extract info from DICOM file and from csv containing experimental info
    # excel实验文件读取，读取文件路径C:\Users\yaesu539\OneDrive - Uppsala universitet\PhD\DATA\MRI\Fe2O3 SiO2 0.1-2mM/info_mri.xlsx
    # pandas DataFrame
    df_info = pd.read_excel(info_csv_fp, sheet_name=parameter, engine='openpyxl')
    # dropna:过滤NA的数据，取出(parameter sheet(T1)的数据值除于1000)
    echo_time = df_info.dropna(subset=[parameter])[[parameter]].values / 1000
    # 数组降维，多维数组变成1维
    echo_time = echo_time.flatten()
    # 获取file_dir路径下所有.dcm文件
    file_paths, filename = get_files(file_dir, ".dcm")
    # pd.DataFrame创建类似excel文件数据列名为filename，每列值为filename数组（从get_files函数可以看出filename是一个数组）
    # filename
    # 1.dcm
    # 2.dcm
    # pd.concat 数据连接，将df_info数据添加多一列，那一列的数值是pd.DataFrame生成的数据  axis指定columns
    df_info = pd.concat([df_info, pd.DataFrame({'filename': filename})], axis=1)
    # 读取manual_keypoints_fp csv文件数据到manual_keypints中，是一个对象数组。具体见函数
    manual_keypoints = read_kp_csv(manual_keypoints_fp)
    print(manual_keypoints)
    # Get the masks according to sample position in MRI (from basefile)
    # 从firstfile获取基础的masks数据
    # first_file=C:\Users\yaesu539\OneDrive - Uppsala universitet\PhD\DATA\MRI\Fe2O3 SiO2 0.1-2mM/DICOM/T1/slice001image001echo001.dcm
    masks = get_masks_from_basefile(first_file, manual_keypoints)

    # Get mean intensities from each sample (based on position of masks from basefile). Input "Yes" or "No" to show/not show names on masks
    samples = get_samples_intensities(file_paths, filename, masks, print_masks)

    # Put sample's intensities into a pandas df
    df_mri = pd.DataFrame(samples)
    # df_mri df_info数据合并，左连接，ind为主键
    df_mri = df_mri.merge(df_info[['ind', 'Concentration']], on='ind', how='left')
    # df_mri df_info数据合并，左连接，filename为主键
    df_mri = df_mri.merge(df_info[['filename', parameter]], on='filename', how='left')
    # display(df_mri)
    df_mri.to_excel(r"C:\Users\yaesu539\Work Folders\Desktop\excelmri.xlsx")

    # Plot data and get r2/r1/r2*
    all_r, Fe_concentration = graph_echotime_vs_intensity_ordered(df_mri, echo_time)
    graph_r_vs_concentration(Fe_concentration, all_r)

