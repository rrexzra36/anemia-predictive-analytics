# Laporan Proyek Machine Learning - Reyhan Ezra Bimantara

## Domain Proyek

Anemia merupakan kondisi kesehatan yang ditandai dengan rendahnya kadar hemoglobin dalam darah, sehingga tubuh tidak mendapatkan cukup oksigen yang dibutuhkan untuk fungsi optimal. Kondisi ini menjadi perhatian kesehatan masyarakat global, terutama di negara berkembang, karena prevalensinya yang tinggi dan dampaknya terhadap produktivitas serta kualitas hidup. Predictive analysis atau analisis prediktif menjadi salah satu pendekatan yang efektif dalam mendeteksi risiko anemia lebih awal, yang memungkinkan intervensi yang tepat waktu dan lebih efisien. Laporan ini berfokus pada penerapan teknik predictive analysis untuk memprediksi kemungkinan seseorang menderita anemia berdasarkan data yang tersedia, serta memberikan wawasan bagi pengambilan keputusan dalam upaya pencegahan dan penanganan anemia secara lebih terarah.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding

### Problem Statements

Berdasarkan penjelasan dalam latar belakang, dapat dirumuskan beberapa permasalahan utama sebagai berikut:

1. Bagaimana cara memprediksi anemia pada manusia dengan menggunakan data sel darah merah melalui penerapan algoritma machine learning?
2. Bagaimana proses pengembangan model machine learning yang mampu memprediksi anemia berdasarkan data sel darah merah?
3. Bagaimana cara mengevaluasi model machine learning yang digunakan untuk memprediksi anemia berdasarkan data sel darah merah?

### Goals

Dari rumusan masalah di atas, penelitian ini bertujuan untuk:

1. Menjelaskan penerapan algoritma machine learning dalam memprediksi anemia berdasarkan data sel darah merah.
2. Menjelaskan proses pengembangan model machine learning yang mampu memprediksi anemia pada manusia.
3. Mengevaluasi kinerja model machine learning dalam memprediksi anemia dengan menggunakan data sel darah merah.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution Statements
Untuk mencapai tujuan yang ditetapkan, penelitian ini mengadopsi beberapa langkah berikut:

1. Melakukan preprocessing data dan exploratory data analysis (EDA) pada dataset, termasuk memvisualisasikan hubungan antara fitur untuk mendapatkan insight yang relevan.
2. Mengembangkan model prediksi menggunakan beberapa algoritma klasifikasi yang sesuai dengan dataset. Jika hasil evaluasi awal kurang memuaskan, model baseline akan ditingkatkan melalui hyperparameter tuning. Algoritma yang digunakan mencakup:
    - **K-Nearest Neighbors (KNN)**, algoritma berbasis instance-based learning yang sederhana namun efektif. Ketika diberikan data baru untuk diklasifikasikan, algoritma ini membandingkan data tersebut dengan tetangga terdekatnya berdasarkan jarak (biasanya menggunakan jarak Euclidean). KNN mengklasifikasikan data baru berdasarkan mayoritas label dari k tetangga terdekatnya. Algoritma ini cocok untuk dataset yang tidak terlalu besar karena proses komputasinya bisa lambat pada dataset besar.
    - **Support Vector Machine (SVM)**, algoritma supervised learning yang mencari hyperplane terbaik untuk memisahkan data dari dua kelas atau lebih. Tujuan SVM adalah menemukan garis batas atau hyperplane yang memaksimalkan margin antara dua kelas, sehingga data dari kelas yang berbeda terpisah sejauh mungkin. SVM bekerja baik dengan dataset linear dan non-linear, di mana untuk dataset non-linear, SVM menggunakan teknik kernel trick untuk memetakan data ke dimensi yang lebih tinggi agar dapat dipisahkan.
    - **Random Forest**, algoritma *ensemble learning* yang menggabungkan banyak decision tree untuk membuat prediksi yang lebih akurat dan stabil. Algoritma ini bekerja dengan membuat banyak decision tree dari subset data yang berbeda dan fitur yang dipilih secara acak. Setiap tree memberikan prediksinya, dan Random Forest mengambil keputusan akhir berdasarkan voting mayoritas dari semua tree tersebut. Karena menggabungkan banyak pohon, Random Forest cenderung lebih tahan terhadap overfitting dibandingkan decision tree tunggal.
3. Melakukan evaluasi kinerja model menggunakan berbagai metrik evaluasi untuk menilai performa dari model yang dikembangkan.

## Data Understanding
Dataset yang digunakan dalam proyek machine learning ini merupakan dataset anemia yang terdiri dari 1421 entri data atau record. Dataset ini bersifat open-source, yang berarti tersedia secara bebas untuk digunakan oleh publik, dan telah dipublikasikan oleh Biswa Ranjan Rao melalui platform Kaggle dengan judul [Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset). Topik utama yang diusung oleh dataset ini adalah kesehatan, khususnya terkait kondisi anemia. Format file dataset tersebut adalah CSV (comma-separated values), yaitu format yang umum digunakan untuk penyimpanan data tabular karena memungkinkan penyusunan data dalam bentuk baris dan kolom. Dengan ukuran file sebesar 34.63 kB, dataset ini relatif ringan dan mudah diakses. Informasi yang terkandung di dalamnya mencakup berbagai fitur yang relevan untuk analisis prediktif anemia, sehingga cocok untuk digunakan dalam pengembangan dan pengujian model machine learning. Penggunaan dataset ini memungkinkan proyek ini untuk memanfaatkan data nyata dalam upaya memprediksi kondisi anemia pada manusia dengan lebih akurat dan efisien.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Gender : jenis kelamin responden (0 = Laki-laki, 1 = Perempuan).
- MCH : *Mean Corpuscular Hemoglobin*, rata-rata massa hemoglobin per sel darah merah dalam sampel darah.
- MCHC : *Mean Corpuscular Hemoglobin Concentration*, konsentrasi rata-rata hemoglobin dalam satu sel darah merah.
- MCV : *Mean Corpuscular Volume*, perhitungan ukuran rata-rata sel darah merah.
- Results : Status penyakit anemia (0 = Negatif Anemia, 1 = Positif Anemia)

## Data Preparation
Teknik yang digunakan pada notebook untuk data Data Preparation yaitu:
1. One-Hot Encoding
  One-Hot Encoding adalah metode yang populer digunakan untuk mengonversi data kategorikal menjadi data numerik dengan format biner, yaitu 0 dan 1. Proses encoding sangat penting dalam machine learning karena sebagian besar algoritma bekerja lebih optimal dengan data numerik daripada kategorikal. Melalui encoding, data kategorikal diubah sehingga dapat diproses dengan baik oleh model machine learning. Dalam proyek ini, proses encoding dilakukan secara manual melalui fungsi khusus yang dikembangkan untuk mengonversi nilai-nilai kategorikal menjadi numerik. Hal ini memungkinkan karena saya sudah memahami dan mengetahui nilai-nilai dari fitur yang akan diencoding. Fitur yang memerlukan encoding dalam proyek ini adalah "Gender" dan "Result," yang dikonversi ke dalam bentuk numerik agar algoritma klasifikasi dapat bekerja lebih efektif.
2. Data Splitting
  Tahap ini bertujuan untuk membagi dataset menjadi dua bagian, yaitu data latih (*train*) dan data uji (*test*). Pembagian ini penting untuk memastikan bahwa model machine learning tidak hanya dilatih tetapi juga dievaluasi kinerjanya pada data yang belum pernah dilihat sebelumnya. Dalam proyek ini, dataset dibagi dengan proporsi 80% untuk data latih dan 20% untuk data uji. Dengan demikian, 427 data digunakan untuk melatih model, sementara 107 data sisanya digunakan untuk pengujian. Proses pembagian ini dilakukan menggunakan fungsi train_test_split() yang tersedia di library sklearn. Pemisahan data ini bertujuan untuk mengevaluasi akurasi model dan melihat bagaimana performanya pada data yang belum pernah digunakan selama pelatihan.

  3. Feature Scaling (Standarisasi)
  Feature scaling bertujuan untuk menormalisasi range dari setiap fitur data sehingga semua fitur berada pada skala yang sama. Jika proses scaling ini tidak dilakukan, model machine learning cenderung lebih terpengaruh oleh fitur dengan nilai yang lebih besar, dan fitur dengan nilai lebih kecil mungkin memiliki pengaruh yang lebih sedikit dalam hasil prediksi. Dalam proyek ini, metode yang digunakan adalah standarisasi, karena dataset memiliki distribusi data yang mendekati normal, dan standarisasi lebih cocok digunakan dalam kasus ini. Proses standarisasi dilakukan dengan memanfaatkan fungsi StandardScaler() dari sklearn. Fungsi ini bekerja dengan cara mengurangi nilai rata-rata (mean) dari setiap fitur dan membaginya dengan standar deviasi, sehingga setiap fitur memiliki rata-rata nol dan varian yang sama. Dengan cara ini, semua fitur memiliki skala yang seragam, memungkinkan model untuk melakukan prediksi yang lebih akurat dan seimbang.

  4. Handling Imbalanced Class
  Ketidakseimbangan kelas (*imbalanced class*) dalam dataset sering menjadi masalah besar, terutama dalam algoritma klasifikasi. Saat proporsi kelas tidak seimbang, model machine learning akan cenderung mengklasifikasikan data ke kelas yang dominan (*majority class*) daripada kelas yang lebih sedikit (*minority class*). Ini bisa menjadi risiko serius, terutama dalam bidang kesehatan, di mana kesalahan dalam prediksi dapat berakibat fatal. Dalam proyek ini, terdapat ketidakseimbangan kelas pada dataset, sehingga teknik **Synthetic Minority Over-sampling Technique** (SMOTE) digunakan untuk menangani masalah ini. SMOTE adalah metode oversampling yang mensintesis sampel baru dari kelas minoritas untuk menyeimbangkan distribusi data. Dengan cara ini, model mendapatkan distribusi data yang lebih seimbang, sehingga dapat mengurangi bias terhadap kelas mayoritas dan memberikan hasil prediksi yang lebih akurat dan adil.

## Modeling
Pada proyek ini, algoritma machine learning yang diterapkan mencakup beberapa metode populer, yaitu `K-Nearest Neighbor`, `Support Vector Machine`, dan `Random Forest`. Setiap algoritma ini dipilih karena kemampuannya yang berbeda-beda dalam menangani masalah klasifikasi, sehingga diharapkan dapat memberikan hasil prediksi yang optimal. Masing-masing algoritma memiliki keunggulan tersendiri dalam menganalisis data dan menghasilkan model yang akurat untuk memprediksi kondisi anemia berdasarkan data yang tersedia.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
