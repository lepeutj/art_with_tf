import tkinter as tk
from tkinter import filedialog
import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()



class ImageSelector:
    def __init__(self, master):
        self.master = master
        master.title("Image Selector")

        self.image1_path = ""
        self.image2_path = ""

        self.image1_label = tk.Label(master, text="Image : None")
        self.image1_label.pack()

        self.image2_label = tk.Label(master, text="Style: None")
        self.image2_label.pack()

        self.select_image1_button = tk.Button(master, text="Select Image 1", command=self.select_image1)
        self.select_image1_button.pack()

        self.select_image2_button = tk.Button(master, text="Select Image 2", command=self.select_image2)
        self.select_image2_button.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start)
        self.start_button.pack()

    def select_image1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if file_path:
            self.image1_path = file_path
            self.image1_label.config(text=f"Image : {file_path}")

    def select_image2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if file_path:
            self.image2_path = file_path
            self.image2_label.config(text=f"Style: {file_path}")

    def start(self):
        if self.image1_path and self.image2_path:
            output_image_size = 384  

            content_path = self.image1_path
            style_image_url = self.image2_path

            # The content image size can be arbitrary.
            content_img_size = (output_image_size, output_image_size)
            # The style prediction model was trained with image size 256 and it's the 
            # recommended image size for the style image (though, other sizes work as 
            # well but will lead to different results).
            style_img_size = (256, 256)  # Recommended to keep it at 256.

            content_image = load_image(content_path, content_img_size)

            style_image = load_image(style_image_url, style_img_size)
            style_image = tf.nn.avg_pool(style_image, ksize=[2,2], strides=[2,2], padding='SAME')
            #show_n([content_image, style_image], ['Content image', 'Style image'])
            outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
            stylized_image = outputs[0]
            show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
            self.master.destroy()
        else:
            tk.messagebox.showwarning("Warning", "Please select two images.")



root = tk.Tk()
my_gui = ImageSelector(root)
root.mainloop()
