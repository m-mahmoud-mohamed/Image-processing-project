import numpy as np
import gradio as gr





import tkinter as tk
from tkinter import filedialog, ttk
import cv2 as cv2

import matplotlib.pyplot as plt
from tkinter.ttk import *
import numpy as np
import random

from scipy.ndimage import median_filter
from skimage.util import random_noise
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fftpack import fft, fft2, fftshift, ifftshift, ifft2

point_matrix = []


def onclick(event):
    if len(point_matrix) >= 2:
        return
    corr = (event.xdata, event.ydata)
    point_matrix.append(corr)
    print(corr)


class ImageProcessorGUI():
    def __init__(self):

        self.image = None

        self.noisyImage = None
        self.img_withoutnoise = None
        self.mask_denoised_image = None
        self.fftimage = None
        self.mask_window = None
        self.s_p_noisyImage=None
        self.fft_shifted=None
        self.gray_image = None




    def open_image(self,x):

        self.image = x
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        #self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        return self.image


    def calculate_histogram(self,x):
        print(x)
        if isinstance(x, np.ndarray):
            hist = cv2.calcHist([x], [0], None, [256], [0, 256])
            f = plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(hist)
            return f
        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        f = plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(hist)
        # show the plotting graph of an image
        return f

    def equalize_histogram(self,x):
        if isinstance(x, np.ndarray):
            equalized_img = cv2.equalizeHist(x)
            equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
            fig = plt.figure(figsize=(10, 7))

            fig.add_subplot(1, 2, 1)
            plt.imshow(equalized_img, 'gray')
            plt.title("Equalized image")

            fig.add_subplot(1, 2, 2)
            plt.plot(equalized_hist)
            plt.title("Equalized histogram")
            return fig
        equalized_img = cv2.equalizeHist(self.gray_image)
        equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
        fig = plt.figure(figsize=(10, 7))

        fig.add_subplot(1, 2, 1)
        plt.imshow(equalized_img, 'gray')
        plt.title("Equalized image")

        fig.add_subplot(1, 2, 2)
        plt.plot(equalized_hist)
        plt.title("Equalized histogram")
        return fig


    def apply_sopel(self,k_size,x):
        if isinstance(x, np.ndarray):
            gradient_x = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=k_size)
            gradient_y = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=k_size)

            # Calculate the gradient magnitude
            gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

            # Normalize the gradient magnitude
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            fig = plt.figure(figsize=(15, 10))

            fig.add_subplot(1, 4, 1)
            plt.imshow(x, 'gray')
            plt.title("original image")

            fig.add_subplot(1, 4, 2)
            plt.imshow(gradient_x, "gray")
            plt.title("sobel x ")

            fig.add_subplot(1, 4, 3)
            plt.imshow(gradient_y, 'gray')
            plt.title("sobel y")

            fig.add_subplot(1, 4, 4)
            plt.imshow(gradient_magnitude, 'gray')
            plt.title("magnitude image")
            return fig
        gradient_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=k_size)
        gradient_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=k_size)

        # Calculate the gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Normalize the gradient magnitude
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


        fig = plt.figure(figsize=(15, 10))

        fig.add_subplot(1, 4, 1)
        plt.imshow(self.image, 'gray')
        plt.title("original image")

        fig.add_subplot(1, 4, 2)
        plt.imshow(gradient_x, "gray")
        plt.title("sobel x ")

        fig.add_subplot(1, 4, 3)
        plt.imshow(gradient_y, 'gray')
        plt.title("sobel y")

        fig.add_subplot(1, 4, 4)
        plt.imshow(gradient_magnitude, 'gray')
        plt.title("magnitude image")
        return fig

    def apply_laplace(self,x):
        if isinstance(x, np.ndarray):
            kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])
            LaplacianImage = cv2.filter2D(src=x, ddepth=-1, kernel=kernel)
            c = -1
            g = x + c * LaplacianImage
            g = np.clip(g, 0, 255)
            fig = plt.figure(figsize=(10, 7))

            fig.add_subplot(1, 3, 1)
            plt.imshow(x, 'gray')
            plt.title("Original image")

            fig.add_subplot(1, 3, 2)
            plt.imshow(LaplacianImage, 'gray')
            plt.title("Laplacian image")

            fig.add_subplot(1, 3, 3)
            plt.imshow(g, 'gray')
            plt.title("New image")
            return fig
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
        LaplacianImage = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
        c = -1
        g = self.image + c * LaplacianImage
        g = np.clip(g, 0, 255)
        fig = plt.figure(figsize=(10, 7))

        fig.add_subplot(1, 3, 1)
        plt.imshow(self.image, 'gray')
        plt.title("Original image")

        fig.add_subplot(1, 3, 2)
        plt.imshow(LaplacianImage, 'gray')
        plt.title("Laplacian image")

        fig.add_subplot(1, 3, 3)
        plt.imshow(g, 'gray')
        plt.title("New image")
        return fig

    def show_fft(self,x):
        if isinstance(x, np.ndarray):
            x=cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(x)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(magnitude_spectrum, 'gray')
            plt.title("fourier image")
            return fig
        f = np.fft.fft2(self.gray_image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        fig = plt.figure(figsize=(5, 5), dpi=100)
        plt.imshow(magnitude_spectrum, 'gray')
        plt.title("fourier image")
        return fig




    def apply_sp_noise(self,sp_amount,x):
        if isinstance(x, np.ndarray):
            xs_p_noisyImage = random_noise(x, mode="s&p", amount=sp_amount)
            return xs_p_noisyImage

        self.s_p_noisyImage = random_noise(self.image, mode="s&p", amount=sp_amount)

        return self.s_p_noisyImage

    def apply_periodic_noise(self,img,frequency,amplitude):
        if isinstance(img, np.ndarray):
            img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            height, width = img.shape[:2]
            x = np.arange(width) / width
            y = np.arange(height) / height
            X, Y = np.meshgrid(x, y)
            noise = amplitude * np.sin(2 * np.pi * (frequency * X + frequency * Y))
            noisy_image = np.clip(img + noise, 0, 255).astype(np.uint8)

            return noisy_image


        height, width = self.gray_image.shape[:2]
        x = np.arange(width) / width
        y = np.arange(height) / height
        X, Y = np.meshgrid(x, y)
        noise = amplitude * np.sin(2 * np.pi * (frequency * X + frequency * Y))
        self.noisyImage = np.clip(self.gray_image + noise, 0, 255).astype(np.uint8)

        return self.noisyImage

    def remove_sp(self,x):
        if isinstance(x, np.ndarray):
            x_withoutnoise = median_filter(x, 3)
            return x_withoutnoise
        self.img_withoutnoise = median_filter(self.s_p_noisyImage,3)

        return self.img_withoutnoise


    def mask(self):
        print(self.gray_image.shape)
        spectrum = fft2(self.gray_image)
        spectrum = fftshift(spectrum)
        noise_filter = np.ones(shape=self.fftimage.shape)
        point1 = point_matrix[0]
        point2 = point_matrix[1]
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]

        noise_filter[int(y1)-5:int(y1)+5, int(x1)-5:int(x1)+5] = 0
        noise_filter[int(y2)-5:int(y2)+5, int(x2)-5:int(x2)+5] = 0


        self.mask_denoised_image = ifft2(fftshift(spectrum * noise_filter))

        self.mask_denoised_image = np.abs(self.mask_denoised_image)

        print(self.mask_denoised_image)
        print(noise_filter)
        print(self.noisyImage)
        return self.mask_denoised_image

    def show_denoised_image(self):
        plt.imshow(self.mask_denoised_image, "gray")
        plt.title("denoised image ")
        plt.show()

    def mask_window(self):

        img = self.noisyImage
        spectrum = fft2(self.gray_image)
        self.fftimage = fftshift(spectrum)

        fig = plt.figure(1, figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(img, 'gray')
        plt.title("image")

        fig.add_subplot(1, 2, 2)
        plt.imshow(self.fftimage, "gray")
        plt.title("fourier image ")

        return fig


    def gaussian_band_reject_filter(self,X):
        print(X)
        if isinstance(X, np.ndarray):
            X=cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(X)
            fshift = np.fft.fftshift(f)


            center_freq = 20
            bandwidth = 30
            order = 2
            rows, cols = X.shape
            crow, ccol = rows // 2, cols // 2
            x = np.linspace(-ccol, cols - ccol - 1, cols)
            y = np.linspace(-crow, rows - crow - 1, rows)
            xv, yv = np.meshgrid(x, y)
            d = np.sqrt(xv ** 2 + yv ** 2)
            mask = 1 / (1 + ((d - center_freq) / bandwidth) ** (2 * order))

            # Apply the mask to the Fourier transformed image
            fshift_filtered = fshift * mask

            # Perform inverse Fourier transform
            f_ishift = np.fft.ifftshift(fshift_filtered)
            filtered_image = np.fft.ifft2(f_ishift)
            xband_reject_image = np.abs(filtered_image)
            xband_reject_image = cv2.normalize(xband_reject_image, None, 0, 255, cv2.NORM_MINMAX)
            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(1, 2, 1)
            plt.imshow(X, 'gray')
            plt.title("noisy image")

            fig.add_subplot(1, 2, 2)
            plt.imshow(xband_reject_image, "gray")
            plt.title("image after gaussian band reject")
            return fig
        # Perform Fourier transform
        f = np.fft.fft2(self.noisyImage)
        fshift = np.fft.fftshift(f)


        center_freq = 20
        bandwidth=30
        order=2
        rows, cols = self.noisyImage.shape
        crow, ccol = rows // 2, cols // 2
        x = np.linspace(-ccol, cols - ccol - 1, cols)
        y = np.linspace(-crow, rows - crow - 1, rows)
        xv, yv = np.meshgrid(x, y)
        d = np.sqrt(xv ** 2 + yv ** 2)
        mask = 1 / (1 + ((d - center_freq) / bandwidth) ** (2 * order))

        # Apply the mask to the Fourier transformed image
        fshift_filtered = fshift * mask

        # Perform inverse Fourier transform
        f_ishift = np.fft.ifftshift(fshift_filtered)
        filtered_image = np.fft.ifft2(f_ishift)
        self.band_reject_image = np.abs(filtered_image)
        self.band_reject_image = cv2.normalize(self.band_reject_image, None, 0, 255, cv2.NORM_MINMAX)
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.noisyImage, 'gray')
        plt.title("noisy image")

        fig.add_subplot(1, 2, 2)
        plt.imshow(self.band_reject_image, "gray")
        plt.title("image after gaussian band reject")


        return fig






app = ImageProcessorGUI()

################################################
with gr.Blocks() as demo:
    gr.Markdown("Image Processing Project")

    with gr.Tab("Upload Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        upload_image_button = gr.Button("upload image")
        ####################
    with gr.Tab("calculate histogram"):
        with gr.Row():
            h_image_input = gr.Image()
            hist = gr.Plot()
        calc_hist_button = gr.Button("calculate histogram")
        ####################
    with gr.Tab("Equalize histogram"):
        with gr.Row():
            eqh_image_input = gr.Image()
            equalized_hist = gr.Plot()
        equalize_hist_button = gr.Button("calculate histogram")
        ####################

    with gr.Tab("Apply sopel"):
        with gr.Row():
            s_image_input = gr.Image()
            kernelsize=gr.inputs.Slider(1,9,2,label="Kernel Size")
            sopel_image = gr.Plot()
        apply_sopel_button = gr.Button("Apply Sopel")
        ####################

    with gr.Tab("Apply Laplace"):
        with gr.Row():
            l_image_input = gr.Image()
            laplace_image = gr.Plot()
        apply_laplace_button = gr.Button("Apply laplace")
        ####################

    with gr.Tab("Apply Fourier Transform"):
        with gr.Row():
            f_image_input = gr.Image()
            fft_image = gr.Plot()
        apply_fourier_button = gr.Button("Apply Fourier Transform")
        ####################

    with gr.Tab("Apply Salt & Pepper Noise"):
        with gr.Row():
            sp_image_input = gr.Image()
            sp_noise_amount=gr.inputs.Slider(0,1,0.1,label="Amount of Salt & Pepper noise")
            sp_image = gr.Image()
        apply_sp_button = gr.Button("Apply Salt & Pepper Noise")
        ####################

    with gr.Tab("Apply Periodic Noise"):
        with gr.Row():
            p_image_input = gr.Image()
            frequency = gr.inputs.Slider(0, 100, 1, label="Frequency")
            amplitude = gr.inputs.Slider(0, 100, 1, label="Amplitude")
            periodic_image = gr.Image()
        apply_periodic_noise_button = gr.Button("Apply Periodic Noise")
        ####################

    with gr.Tab("Remove salt & pepper Noise"):
        with gr.Row():
            rsp_image_input = gr.Image()
            r_sp_image = gr.Image()
        remove_sp_noise_button = gr.Button("Remove salt & pepper Noise")
        ####################

    with gr.Tab("Remove periodic noise "):
        with gr.Row():
            b_r_image_input = gr.Image()
            band_image = gr.Plot()
        band_image_button = gr.Button("Remove periodic noise")
        ####################


    with gr.Accordion("Done by"):
        gr.Markdown("Mahmoud Mohamed Mahmoud")
        gr.Markdown("Mohamed Mamdouh")
        gr.Markdown("Begad Soliman")

    upload_image_button.click(app.open_image, inputs=image_input, outputs=image_output)
    calc_hist_button.click(app.calculate_histogram,inputs=h_image_input, outputs=hist)
    equalize_hist_button.click(app.equalize_histogram,inputs=eqh_image_input, outputs=equalized_hist)
    apply_sopel_button.click(app.apply_sopel, inputs=[kernelsize,s_image_input], outputs=sopel_image)
    apply_laplace_button.click(app.apply_laplace,inputs=l_image_input, outputs=laplace_image)
    apply_fourier_button.click(app.show_fft,inputs=f_image_input, outputs=fft_image)
    apply_sp_button.click(app.apply_sp_noise, inputs=[sp_noise_amount,sp_image_input],outputs=sp_image)
    apply_periodic_noise_button.click(app.apply_periodic_noise,inputs=[p_image_input,frequency,amplitude],outputs=periodic_image)
    remove_sp_noise_button.click(app.remove_sp,inputs=rsp_image_input,outputs=r_sp_image)
    band_image_button.click(app.gaussian_band_reject_filter,inputs=b_r_image_input,outputs=band_image)



demo.launch()