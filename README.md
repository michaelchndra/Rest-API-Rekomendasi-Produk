> [!IMPORTANT]  
> Project ini masih tahap pengembangan lebih lanjut. <br>
> Jika menemukan bug silahkan laporkan di Issues repositori ini. <br>
---


<br/>
<p align="center">
  <h3 align="center">AI Rekomendasi Produk</h3>

  <p align="center">
     Project Massive Internal Kelompok 1 IBM-Advance AI dari Infinite Learning.
    <br/>
    <br/>
  </p>
</p>



## Table Of Contents

* [Description](#description)
* [Struktur Direktori](#struktur-direktori)
* [File dan Fungsinya](#file-dan-fungsinya)
* [Cara Instalasi dan Menjalankan Aplikasi](#cara-instalasi-dan-menjalankan-aplikasi)
* [Teknologi yang digunakan](#teknologi--tech)

## Description
**Proyek ini adalah sebuah aplikasi web yang memberikan rekomendasi produk berdasarkan input pengguna menggunakan model BERT untuk klasifikasi teks. Data produk diambil dari Tokopedia menggunakan skrip scraping dan model dilatih untuk memberikan rekomendasi produk.**

## Struktur Direktori

```
└── 📁chatbot-rekomendasi
    └── app.py
    └── bert_model.pth
    └── dataset.csv
    └── index.html
    └── README.md
    └── requirements.txt
    └── run.py
    └── scrape.py
```
## File dan Fungsinya


#### `app.py`

File utama untuk aplikasi Flask yang memuat rute untuk mendapatkan rekomendasi produk.

**Fitur Tersedia** <br/>
- Memuat dan memproses dataset
- Melatih model BERT untuk klasifikasi
- Menyimpan dan memuat model terlatih
- Menyediakan API untuk mendapatkan rekomendasi produk berdasarkan input pengguna
python

#### `scrape.py`
Skrip untuk melakukan web scraping dari Tokopedia dan menyimpan data produk ke dalam file CSV **(```dataset.csv```)**.

#### `run.py`
Script ini berfungsi untuk memulai Scraping Data dan Aplikasi Flask .

## Cara Instalasi dan Menjalankan Aplikasi
### Prasyarat
- Python 3.12 atau lebih baru
- Paket Python yang diperlukan (tercantum di requirements.txt)

### Instalasi
1. Clone repositori ini: </br>
```bash
git clone https://github.com/michaelchndra/chatbot-rekomendasi.git
cd chatbot-rekomendasi
```
2. Install Library
```bash
pip install -r requirements.txt
```
3. Jalankan `run.py`
```bash
python run.py
```

### Menggunakan Aplikasi
- Buka file `index.html`.
- Masukkan rekomendasi di input button.
- Klik tombol "Dapatkan Rekomendasi" untuk mendapatkan rekomendasi produk berdasarkan input Anda.

## Teknologi / Tech
### Frontend
- **HTML/CSS**: Struktur dan desain halaman web.
- **JavaScript**: Menangani permintaan AJAX untuk interaksi dinamis tanpa memuat ulang halaman.

### Backend, Framework dan Model
- **Flask**: Mengatur rute API, menerima input pengguna, dan memberikan rekomendasi produk dalam bentuk JSON.
- **Transformers**: Menyediakan pre-trained model BERT yang digunakan untuk klasifikasi teks.
- **PyTorch**: Framework deep learning untuk melatih model BERT.
- **Scikit-learn**: Digunakan untuk memproses dataset, encoding label, dan evaluasi performa model.

### Data Processing
- **Pandas**: Manipulasi dan pembersihan dataset.
- **Requests**: Mengambil data dari Tokopedia.
- **BeautifulSoup**: Parsing data HTML dari hasil scraping.

### Diagram Alur
- **Scraping**: scrape.py mengambil data dari Tokopedia menggunakan requests dan BeautifulSoup, lalu menyimpannya ke dataset.csv.
- **Training**: app.py memuat dataset, memproses teks dengan BERT tokenizer, membagi data dengan train_test_split dari Scikit-learn, melatih model BERT menggunakan PyTorch.
- **Serving**: Aplikasi Flask di app.py melayani permintaan rekomendasi, memuat model terlatih, dan menghasilkan rekomendasi berdasarkan input pengguna.
- **Frontend**: Halaman HTML di index.html menerima input pengguna, mengirim permintaan ke backend Flask, dan menampilkan hasil rekomendasi.

## Thanks to

[**InfiniteLearning**](https://www.infinitelearning.id)
