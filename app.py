from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from compress import compress_image_dct, compress_image_dft, compress_audio_dct, compress_audio_dft, compress_video_dct, compress_video_dft

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['COMPRESSED_FOLDER'] = 'compressed/'

def get_file_size(file_path):
    return os.path.getsize(file_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        compression_type = request.form['compression_type']
        if compression_type == 'image_dct':
            output_filename = 'compressed_dct_' + filename
            output_path = os.path.join(app.config['COMPRESSED_FOLDER'], output_filename)
            compress_image_dct(filepath, output_path, 10, 8)
        elif compression_type == 'image_dft':
            output_filename = 'compressed_dft_' + filename
            output_path = os.path.join(app.config['COMPRESSED_FOLDER'], output_filename)
            compress_image_dft(filepath, 100 ,output_path)
        elif compression_type == 'audio_dct':
            output_filename = 'compressed_dct_' + filename
            output_path = os.path.join(app.config['COMPRESSED_FOLDER'], output_filename)
            compress_audio_dct(filepath, output_path)
        elif compression_type == 'audio_dft':
            output_filename = 'compressed_dft_' + filename
            output_path = os.path.join(app.config['COMPRESSED_FOLDER'], output_filename)
            compress_audio_dft(filepath, output_path)
        elif compression_type == 'video_dct':
            output_filename = 'compressed_dct_' + filename
            output_path = os.path.join(app.config['COMPRESSED_FOLDER'], output_filename)
            compress_video_dct(filepath, output_path)
        elif compression_type == 'video_dft':
            output_filename = 'compressed_dft_' + filename
            output_path = os.path.join(app.config['COMPRESSED_FOLDER'], output_filename)
            compress_video_dft(filepath, output_path)

        original_size = get_file_size(filepath)
        compressed_size = get_file_size(output_path)
        
        return render_template('result.html', original_file=filename, compressed_file=output_filename, 
                               original_size=original_size, compressed_size=compressed_size)
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/compressed/<filename>')
def compressed_file(filename):
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename)

@app.route('/download/<filename>')
def download_compressed(filename):
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
