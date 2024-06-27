import numpy as np
from scipy.fftpack import dct
from PIL import Image
import cv2
from pydub import AudioSegment
import skimage
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pydub import AudioSegment
import scipy.fftpack as ft

# Kompresi gambar menggunakan DCT




# overlaying the mask, discarding the high-frequency components
def compress_image_dct(image_path,output_path, C, N):
    # Read image
    img = skimage.io.imread(image_path, as_gray=True)

    # Obtaining a mask through zigzag scanning
    mask = z_scan_mask(C, N)

    # overlaying the mask, discarding the high-frequency components
    img_dct = Compress(img, mask, N)

    plt.imsave(output_path, img_dct, cmap='gray')

def z_scan_mask(C, N):
    mask = np.zeros((N, N))
    start = 0
    mask_m = start
    mask_n = start
    for i in range(C):
        if i == 0:
            mask[mask_m, mask_n] = 1
        else:
            # If even, move upward to the right
            if (mask_m + mask_n) % 2 == 0:
                mask_m -= 1
                mask_n += 1
                # If it exceeds the upper boundary, move downward
                if mask_m < 0:
                    mask_m += 1
                # If it exceeds the right boundary, move left
                if mask_n >= N:
                    mask_n -= 1
            # If odd, move downward to the left
            else:
                mask_m += 1
                mask_n -= 1
                # If it exceeds the lower boundary, move upward
                if mask_m >= N:
                    mask_m -= 1
                # If it exceeds the left boundary, move right
                if mask_n < 0:
                    mask_n += 1
            mask[mask_m, mask_n] = 1
    return mask

def Compress(img, mask, N):
    img_dct = np.zeros((img.shape[0] // N * N, img.shape[1] // N * N))
    for m in range(0, img_dct.shape[0], N):
        for n in range(0, img_dct.shape[1], N):
            block = img[m:m + N, n:n + N]
            # DCT
            coeff = cv2.dct(block)
            # IDCT, but only the parts of the image where the mask has a value of 1 are retained
            iblock = cv2.idct(coeff * mask)
            img_dct[m:m + N, n:n + N] = iblock
    return img_dct
            
# Kompresi gambar menggunakan DFT
def compress_image_dft(input_path, output_path):
    image = Image.open(input_path).convert('L')
    image_data = np.array(image, dtype=np.float32)
    dft_image = np.fft.fft2(image_data)
    dft_image_real = np.real(dft_image)
    compressed_image = np.clip(dft_image_real, 0, 255).astype(np.uint8)
    compressed_image_pil = Image.fromarray(compressed_image)
    compressed_image_pil.save(output_path)

# Kompresi audio menggunakan DCT
def compress_audio_dct(input_path: str, output_path: str):
    
    audio = AudioSegment.from_file(input_path)
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Define block size and calculate the number of blocks
    block_size = 1024
    num_blocks = len(audio_data) // block_size

    # Apply DCT to each block and store the result in a new array
    compressed_data = np.empty((num_blocks, block_size), dtype=np.float32)
    for i in range(num_blocks):
        block = audio_data[i * block_size:(i + 1) * block_size]
        dct_block = dct(block, type=2)
        compressed_data[i] = dct_block

    # Create a new compressed audio object and export it to the output path
    compressed_audio = AudioSegment(
        compressed_data.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    compressed_audio.export(output_path, format="mp3", bitrate="64k")

# Kompresi audio menggunakan DFT
def compress_audio_dft(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    block_size = 1024
    num_blocks = len(audio_data) // block_size
    for i in range(num_blocks):
        block = audio_data[i * block_size:(i + 1) * block_size]
        dft_block = np.fft.fft(block)
        audio_data[i * block_size:(i + 1) * block_size] = dft_block.real
    compressed_audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    compressed_audio.export(output_path, format="mp3", bitrate="64k")

# Fungsi untuk melakukan kompresi video menggunakan metode DCT


def compress_video_dct(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Create a VideoWriter object to save the compressed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the DCT to the grayscale frame
        gray_dct = ft.dct(gray, norm='ortho')

        # Quantize the DCT coefficients (not shown here)
        gray_quant = quantize(gray_dct)

        # Reconstruct the compressed frame
        gray_compressed = ft.idct(gray_quant, norm='ortho')

        # Convert the compressed grayscale frame back to BGR
        frame_compressed = cv2.cvtColor(gray_compressed.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Write the compressed frame to the output video
        out.write(frame_compressed)

    # Release the video objects
    cap.release()
    out.release()

    cv2.destroyAllWindows()

    
    
# Kompresi video menggunakan DFT

def compress_video_dft(input_path, output_path, compression_ratio=0.5):
    """
    Compress a video using DFT and quantization.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to the output compressed video file.
        compression_ratio (float, optional): Compression ratio (0 < ratio < 1). Defaults to 0.5.

    Returns:
        None
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply DFT to the grayscale frame
        dft = dct(dct(gray.T, norm='ortho').T, norm='ortho')

        # Quantize the DFT coefficients
        quantized_dft = quantize(dft, compression_ratio)

        # Reconstruct the compressed frame
        compressed_gray = idct(idct(quantized_dft.T, norm='ortho').T, norm='ortho')

        # Convert the compressed grayscale frame back to BGR
        compressed_frame = cv2.cvtColor(compressed_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        out.write(compressed_frame)

    cap.release()
    out.release()
    
    
def quantize(dct_coeffs, quality=50):
    """
    Quantize the DCT coefficients using a quality factor.

    Args:
        dct_coeffs (numpy array): DCT coefficients to be quantized.
        quality (int, optional): Quality factor (1-100). Defaults to 50.

    Returns:
        numpy array: Quantized DCT coefficients.
    """
    # Define the quantization matrix for the given quality factor
    quant_matrix = get_quantization_matrix(quality)

    # Repeat the quantization matrix to match the shape of the DCT coefficients
    repeat_rows = dct_coeffs.shape[0] // 8 + 1
    repeat_cols = dct_coeffs.shape[1] // 8 + 1
    quant_matrix_repeated = np.repeat(np.repeat(quant_matrix, repeat_rows, axis=0), repeat_cols, axis=1)
    quant_matrix_repeated = quant_matrix_repeated[:dct_coeffs.shape[0], :dct_coeffs.shape[1]]

    # Quantize the DCT coefficients
    quantized_coeffs = np.round(dct_coeffs / quant_matrix_repeated)

    return quantized_coeffs

def get_quantization_matrix(quality):
    """
    Get the quantization matrix for the given quality factor.

    Args:
        quality (int): Quality factor (1-100).

    Returns:
        numpy array: Quantization matrix.
    """
    # Define the default quantization matrix for JPEG
    default_quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Scale the quantization matrix based on the quality factor
    scale = 50 / quality
    quant_matrix = default_quant_matrix * scale

    return quant_matrix