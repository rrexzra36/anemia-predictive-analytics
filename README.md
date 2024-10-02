# Laporan Proyek Machine Learning - Reyhan Ezra Bimantara
<p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/background.png" width="100%"/>
</p>

<p align="center">
  Sumber gambar: <strong><a href="https://www.technologynetworks.com/cell-science/news/3d-bioprinted-blood-vessel-developed-352139">3D-Bioprinted Blood Vessel Developed</a></strong>
</p>

## Domain Proyek

Anemia merupakan kondisi kesehatan yang ditandai dengan rendahnya kadar hemoglobin dalam darah, sehingga tubuh tidak mendapatkan cukup oksigen yang dibutuhkan untuk fungsi optimal. Kondisi ini menjadi perhatian kesehatan masyarakat global, terutama di negara berkembang, karena prevalensinya yang tinggi dan dampaknya terhadap produktivitas serta kualitas hidup. Predictive analysis atau analisis prediktif menjadi salah satu pendekatan yang efektif dalam mendeteksi risiko anemia lebih awal, yang memungkinkan intervensi yang tepat waktu dan lebih efisien. Laporan ini berfokus pada penerapan teknik predictive analysis untuk memprediksi kemungkinan seseorang menderita anemia berdasarkan data yang tersedia, serta memberikan wawasan bagi pengambilan keputusan dalam upaya pencegahan dan penanganan anemia secara lebih terarah.

**Referensi:**
- [Revolutionizing anemia detection: integrative machine learning models and advanced attention mechanisms](https://vciba.springeropen.com/articles/10.1186/s42492-024-00169-4) 
- [Anemia Prediction Using Machine Learning Algorithms](https://link.springer.com/chapter/10.1007/978-3-031-58604-0_20)
- [Machine Learning and Predictive Analytics: Advancing Disease Prevention in Healthcare](https://publications.dlpress.org/index.php/jcha/article/view/16)

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

### Solution Statements
Untuk mencapai tujuan yang ditetapkan, penelitian ini mengadopsi beberapa langkah berikut:

1. Melakukan preprocessing data dan exploratory data analysis (EDA) pada dataset, termasuk memvisualisasikan hubungan antara fitur untuk mendapatkan insight yang relevan.
2. Mengembangkan model prediksi menggunakan beberapa algoritma klasifikasi yang sesuai dengan dataset. Jika hasil evaluasi awal kurang memuaskan, model baseline akan ditingkatkan melalui hyperparameter tuning. Algoritma yang digunakan mencakup:
    - **K-Nearest Neighbors (KNN)**, algoritma berbasis instance-based learning yang sederhana namun efektif. Ketika diberikan data baru untuk diklasifikasikan, algoritma ini membandingkan data tersebut dengan tetangga terdekatnya berdasarkan jarak (biasanya menggunakan jarak Euclidean). KNN mengklasifikasikan data baru berdasarkan mayoritas label dari k tetangga terdekatnya. Algoritma ini cocok untuk dataset yang tidak terlalu besar karena proses komputasinya bisa lambat pada dataset besar.
    - **Support Vector Machine (SVM)**, algoritma supervised learning yang mencari hyperplane terbaik untuk memisahkan data dari dua kelas atau lebih. Tujuan SVM adalah menemukan garis batas atau hyperplane yang memaksimalkan margin antara dua kelas, sehingga data dari kelas yang berbeda terpisah sejauh mungkin. SVM bekerja baik dengan dataset linear dan non-linear, di mana untuk dataset non-linear, SVM menggunakan teknik kernel trick untuk memetakan data ke dimensi yang lebih tinggi agar dapat dipisahkan.
    - **Random Forest**, algoritma *ensemble learning* yang menggabungkan banyak decision tree untuk membuat prediksi yang lebih akurat dan stabil. Algoritma ini bekerja dengan membuat banyak decision tree dari subset data yang berbeda dan fitur yang dipilih secara acak. Setiap tree memberikan prediksinya, dan Random Forest mengambil keputusan akhir berdasarkan voting mayoritas dari semua tree tersebut. Karena menggabungkan banyak pohon, Random Forest cenderung lebih tahan terhadap overfitting dibandingkan decision tree tunggal.
3. Melakukan evaluasi kinerja model menggunakan berbagai metrik evaluasi untuk menilai performa dari model yang dikembangkan.

## Data Understanding
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/dataset.png" />
</p>

Dataset yang digunakan dalam proyek machine learning ini merupakan dataset anemia yang terdiri dari 1421 entri data atau record. Dataset ini bersifat open-source, yang berarti tersedia secara bebas untuk digunakan oleh publik, dan telah dipublikasikan oleh Biswa Ranjan Rao melalui platform Kaggle dengan judul **[Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset)**. Topik utama yang diusung oleh dataset ini adalah kesehatan, khususnya terkait kondisi anemia. Format file dataset tersebut adalah CSV (comma-separated values), yaitu format yang umum digunakan untuk penyimpanan data tabular karena memungkinkan penyusunan data dalam bentuk baris dan kolom. Dengan ukuran file sebesar 34.63 kB, dataset ini relatif ringan dan mudah diakses. Informasi yang terkandung di dalamnya mencakup berbagai fitur yang relevan untuk analisis prediktif anemia, sehingga cocok untuk digunakan dalam pengembangan dan pengujian model machine learning. Penggunaan dataset ini memungkinkan proyek ini untuk memanfaatkan data nyata dalam upaya memprediksi kondisi anemia pada manusia dengan lebih akurat dan efisien.

**Variabel-variabel pada Anemia Dataset adalah sebagai berikut:**
- **Gender** : jenis kelamin responden (0 = Laki-laki, 1 = Perempuan).
- **MCH** : *Mean Corpuscular Hemoglobin*, rata-rata massa hemoglobin per sel darah merah dalam sampel darah.
- **MCHC** : *Mean Corpuscular Hemoglobin Concentration*, konsentrasi rata-rata hemoglobin dalam satu sel darah merah.
- **MCV** : *Mean Corpuscular Volume*, perhitungan ukuran rata-rata sel darah merah.
- **Results** : Status penyakit anemia (0 = Negatif Anemia, 1 = Positif Anemia)

### Hasil Visualisasi dan Analisis Data
  1. **Univariate Analysis**\
  Dari keseluruhan populasi, 46,3% orang terindikasi menderita anemia, sementara 53,7% lainnya tidak mengalami anemia. Dengan demikian, meskipun sedikit lebih dari separuh populasi tidak terpengaruh oleh anemia, hampir setengah dari total populasi masih mengalami kondisi tersebut, menunjukkan prevalensi anemia yang signifikan.
  <p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/uni_analysis.png" width="50%" />
  </p>

  2. **Bivariate Analysis**\
  Berdasarkan analisis boxplot, dapat disimpulkan bahwa terdapat perbedaan signifikan pada kadar hemoglobin berdasarkan jenis kelamin dan hasil tes. Pria secara umum memiliki kadar hemoglobin yang lebih tinggi dibandingkan wanita, baik pada kelompok dengan hasil tes positif maupun negatif. Selain itu, hasil tes juga mempengaruhi kadar hemoglobin, di mana individu dengan hasil negatif cenderung memiliki kadar hemoglobin yang lebih tinggi dibandingkan individu dengan hasil positif. Kelompok wanita dengan hasil positif menunjukkan kadar hemoglobin terendah dibandingkan kelompok lainnya, bahkan terdapat beberapa outlier yang menunjukkan nilai hemoglobin sangat rendah. Secara keseluruhan, pria dengan hasil negatif memiliki rentang kadar hemoglobin yang lebih luas dan lebih tinggi, sementara wanita dengan hasil positif memiliki rentang kadar hemoglobin yang lebih sempit dengan nilai yang lebih rendah.
  <p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/bi_analysis.png" width="50%" />
  </p>


  3. **Multivariate Analysis**\
  Scatter plot di antara variabel-variabel ini memperlihatkan pola-pola hubungan yang tersebar, meskipun untuk beberapa variabel (seperti Hemoglobin dan MCV), terdapat pemisahan yang lebih jelas antara individu yang anemik dan tidak anemik. Pairplot ini membantu mengidentifikasi pola keterkaitan antar variabel serta bagaimana hasil tes anemia mempengaruhi distribusi variabel-variabel tersebut.
  <p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/multi_analysis.png" width="50%" />
  </p>
  
  4. **Outlier & Distribution Analysis**\
  Visualisasi boxplot berguna untuk mendeteksi keberadaan outlier pada setiap fitur. Dari analisis data yang digunakan, tidak terdapat outlier yang teridentifikasi. Selanjutnya, pada histogram distribusi normal, fitur MCH, MCHC, dan MCV menunjukkan pola distribusi yang normal. Di sisi lain, fitur Hemoglobin menunjukkan kecenderungan sedikit miring ke arah kiri (left-skewed).
  <p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/out_boxplot.png" width="50%" />
  </p>
  <p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/out_hist.png" width="50%" />
  </p>


## Data Preparation
Teknik yang digunakan pada notebook untuk data Data Preparation yaitu:
1. **One-Hot Encoding**\
  One-Hot Encoding adalah metode yang populer digunakan untuk mengonversi data kategorikal menjadi data numerik dengan format biner, yaitu 0 dan 1. Proses encoding sangat penting dalam machine learning karena sebagian besar algoritma bekerja lebih optimal dengan data numerik daripada kategorikal. Melalui encoding, data kategorikal diubah sehingga dapat diproses dengan baik oleh model machine learning. Dalam proyek ini, proses encoding dilakukan secara manual melalui fungsi khusus yang dikembangkan untuk mengonversi nilai-nilai kategorikal menjadi numerik. Hal ini memungkinkan karena saya sudah memahami dan mengetahui nilai-nilai dari fitur yang akan diencoding. Fitur yang memerlukan encoding dalam proyek ini adalah "Gender" dan "Result," yang dikonversi ke dalam bentuk numerik agar algoritma klasifikasi dapat bekerja lebih efektif.
2. **Data Splitting**\
  Tahap ini bertujuan untuk membagi dataset menjadi dua bagian, yaitu data latih (*train*) dan data uji (*test*). Pembagian ini penting untuk memastikan bahwa model machine learning tidak hanya dilatih tetapi juga dievaluasi kinerjanya pada data yang belum pernah dilihat sebelumnya. Dalam proyek ini, dataset dibagi dengan proporsi 80% untuk data latih dan 20% untuk data uji. Dengan demikian, 427 data digunakan untuk melatih model, sementara 107 data sisanya digunakan untuk pengujian. Proses pembagian ini dilakukan menggunakan fungsi train_test_split() yang tersedia di library sklearn. Pemisahan data ini bertujuan untuk mengevaluasi akurasi model dan melihat bagaimana performanya pada data yang belum pernah digunakan selama pelatihan.

  3. **Feature Scaling (Standarisasi)**\
  Feature scaling bertujuan untuk menormalisasi range dari setiap fitur data sehingga semua fitur berada pada skala yang sama. Jika proses scaling ini tidak dilakukan, model machine learning cenderung lebih terpengaruh oleh fitur dengan nilai yang lebih besar, dan fitur dengan nilai lebih kecil mungkin memiliki pengaruh yang lebih sedikit dalam hasil prediksi. Dalam proyek ini, metode yang digunakan adalah standarisasi, karena dataset memiliki distribusi data yang mendekati normal, dan standarisasi lebih cocok digunakan dalam kasus ini. Proses standarisasi dilakukan dengan memanfaatkan fungsi StandardScaler() dari sklearn. Fungsi ini bekerja dengan cara mengurangi nilai rata-rata (mean) dari setiap fitur dan membaginya dengan standar deviasi, sehingga setiap fitur memiliki rata-rata nol dan varian yang sama. Dengan cara ini, semua fitur memiliki skala yang seragam, memungkinkan model untuk melakukan prediksi yang lebih akurat dan seimbang.

  4. **Handling Imbalanced Class**\
  Ketidakseimbangan kelas (*imbalanced class*) dalam dataset sering menjadi masalah besar, terutama dalam algoritma klasifikasi. Saat proporsi kelas tidak seimbang, model machine learning akan cenderung mengklasifikasikan data ke kelas yang dominan (*majority class*) daripada kelas yang lebih sedikit (*minority class*). Ini bisa menjadi risiko serius, terutama dalam bidang kesehatan, di mana kesalahan dalam prediksi dapat berakibat fatal. Dalam proyek ini, terdapat ketidakseimbangan kelas pada dataset, sehingga teknik **Synthetic Minority Over-sampling Technique (SMOTE)** digunakan untuk menangani masalah ini. SMOTE adalah metode oversampling yang mensintesis sampel baru dari kelas minoritas untuk menyeimbangkan distribusi data. Dengan cara ini, model mendapatkan distribusi data yang lebih seimbang, sehingga dapat mengurangi bias terhadap kelas mayoritas dan memberikan hasil prediksi yang lebih akurat dan adil.
  <p align="center">
    <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/smote_before.png" alt="Deskripsi Gambar 1" style="width: 300px; height: auto;">
    <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/smote_after.png" alt="Deskripsi Gambar 2" style="width: 300px; height: auto;">
  </p>

## Modeling
Pada proyek ini, algoritma machine learning yang diterapkan mencakup beberapa metode populer, yaitu `K-Nearest Neighbor`, `Support Vector Machine`, dan `Random Forest`. Setiap algoritma ini dipilih karena kemampuannya yang berbeda-beda dalam menangani masalah klasifikasi, sehingga diharapkan dapat memberikan hasil prediksi yang optimal. Masing-masing algoritma memiliki keunggulan tersendiri dalam menganalisis data dan menghasilkan model yang akurat untuk memprediksi kondisi anemia berdasarkan data yang tersedia.

### **Algoritma K-Nearest Neighbor (KNN)** <br>

<p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/knn.png" width=80%" />
</p>

KNN adalah salah satu algoritma klasifikasi yang sederhana namun sangat populer digunakan dalam machine learning. KNN bekerja dengan menentukan class dari data baru berdasarkan sejumlah K data terdekat yang dijadikan acuan. Algoritma ini menggunakan jarak (similarity) sebagai dasar untuk membandingkan setiap data, di mana K tetangga terdekat dari data baru dipilih untuk menentukan class yang paling sesuai. Dalam proyek ini, proses pemodelan menggunakan KNN dilakukan dengan memanfaatkan modul KNeighborsClassifier() yang tersedia di library scikit-learn. Parameter yang digunakan adalah `n_neighbors = 10`, yang berarti model akan menggunakan 10 data terdekat sebagai acuan dalam menentukan class pada proses klasifikasi. 

Untuk menghitung similarity, proyek ini menggunakan minkowski distance sebagai metrik jarak. **Minkowski distance** merupakan generalisasi dari beberapa jenis jarak lainnya, seperti Euclidean dan Manhattan distance. Perbedaan utama minkowski terletak pada adanya parameter p (pangkat) yang dapat diatur. Jika `p=1`, maka jarak yang dihitung adalah Manhattan distance, sedangkan jika `p=2`, maka jaraknya adalah Euclidean distance. Minkowski menghitung jarak antara dua vektor data, dan variasi nilai p ini memberikan fleksibilitas lebih dalam menentukan cara pengukuran jarak antar data pada klasifikasi KNN.

  <table border="1">
  <tr>
    <th>Kelebihan KNN</th>
    <th>Kelemahan KNN</th>
  </tr>
  <tr>
    <td>Mudah diimplementasikan dan dipahami karena konsepnya sederhana.</td>
    <td>Sangat bergantung pada ukuran K yang dipilih, sehingga pemilihan K yang salah dapat mempengaruhi hasil.</td>
  </tr>
  <tr>
    <td>Tidak memerlukan pelatihan model secara eksplisit, karena bekerja berdasarkan instance-based learning.</td>
    <td>Tidak efisien untuk dataset besar, karena harus menghitung jarak setiap data dalam dataset secara berulang.</td>
  </tr>
  <tr>
    <td>Fleksibel dalam penggunaan metrik jarak yang berbeda seperti Euclidean, Manhattan, dan Minkowski.</td>
    <td>Sensitif terhadap data yang tidak relevan atau fitur yang memiliki skala berbeda tanpa normalisasi.</td>
  </tr>
  <tr>
    <td>Dapat digunakan baik untuk klasifikasi maupun regresi.</td>
    <td>Rentan terhadap outlier karena tetangga terdekat bisa terpengaruh oleh data yang tidak representatif.</td>
  </tr>
  <tr>
    <td>Kinerja bagus pada dataset dengan distribusi yang jelas atau data yang terpisah dengan baik.</td>
    <td>Sulit menangani data yang tidak seimbang karena model cenderung lebih memprioritaskan majority class.</td>
  </tr>
</table>

### Algoritma Support Vector Machine (SVM)  <br>

<p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/svm.jpg" width="80%" />
</p>

berfungsi untuk menemukan *hyperplane* terbaik dengan cara memaksimalkan jarak antara kelas-kelas. *Hyperplane* adalah sebuah fungsi yang berfungsi sebagai pemisah antara kelas. Dalam ruang dua dimensi, fungsi yang digunakan untuk klasifikasi antar kelas disebut sebagai garis, sementara fungsi yang digunakan untuk klasifikasi dalam tiga dimensi disebut sebagai bidang. Demikian pula, fungsi yang digunakan untuk klasifikasi dalam ruang berdimensi lebih tinggi disebut *hyperplane*.

Pada tahap pemodelan, algoritma Support Vector Machine (SVM) yang digunakan menerapkan metode kernel dan menerima semua vektor input yang disediakan dalam data pelatihan dengan menggunakan parameter RBF sebagai teknik kernel-nya. Kernel ini dikenal memiliki kinerja yang baik dan menghasilkan nilai kesalahan yang rendah pada hasil pelatihan. Fungsi kernel RBF dinyatakan sebagai berikut: 
  <p align="center">
    <img src="https://latex.codecogs.com/svg.image?{\color{White}K(x,x_i{})=(-\gamma\cdot\sum(x,x_i{})^{2}})">

  </p>
  <table border="1">
  <tr>
    <th>Kelebihan SVM</th>
    <th>Kekurangan SVM</th>
  </tr>
  <tr>
    <td>Mampu menangani data dengan dimensi yang tinggi dan efektif dalam ruang fitur yang besar.</td>
    <td>Waktu pelatihan bisa sangat lama, terutama pada dataset yang besar dan kompleks.</td>
  </tr>
  <tr>
    <td>Memberikan hasil klasifikasi yang baik meskipun ada noise dalam data.</td>
    <td>Kurang efektif pada dataset yang sangat besar, karena dapat mengalami masalah dalam hal komputasi.</td>
  </tr>
  <tr>
    <td>Dapat digunakan untuk klasifikasi dan regresi dengan fleksibilitas tinggi melalui penggunaan berbagai jenis kernel.</td>
    <td>Memilih parameter kernel dan pengaturan hyperparameter yang tepat dapat menjadi sulit dan mempengaruhi hasil.</td>
  </tr>
  <tr>
    <td>Menawarkan kontrol yang baik terhadap margin, sehingga lebih tahan terhadap overfitting pada data pelatihan.</td>
    <td>Sensitif terhadap pemilihan kernel, yang dapat berdampak signifikan pada performa model.</td>
  </tr>
  <tr>
    <td>SVM sangat efektif dalam situasi di mana jumlah dimensi fitur lebih besar daripada jumlah sampel.</td>
    <td>Interpretasi model lebih sulit dibandingkan dengan algoritma yang lebih sederhana seperti regresi logistik.</td>
  </tr>
</table>

### Algoritma Random Forest (RF)<br>

<p align="center">
      <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/random-forest.jpg" width="80%" />
</p>

Memanfaatkan kombinasi beberapa pohon keputusan (*decision tree*) untuk menghasilkan prediksi yang lebih akurat. Prinsip dasar dari Random Forest adalah bahwa beberapa pohon keputusan yang tidak saling berkorelasi akan berfungsi lebih baik sebagai kelompok dibandingkan jika mereka beroperasi secara individual. Saat Random Forest digunakan sebagai pengklasifikasi, setiap pohon keputusan memberikan satu suara, dan setiap pohon keputusan dapat menghasilkan jawaban yang sama atau berbeda. Sebagai contoh, pohon keputusan A, B, E, dan F mungkin memprediksi hasil 1, sementara pohon keputusan C dan D memprediksi hasil 0. Dengan banyaknya kemungkinan jawaban dari pohon keputusan dan risiko bias yang tinggi, Random Forest mengambil prediksi dari sejumlah pohon keputusan berdasarkan suara mayoritas, sehingga menghasilkan prediksi yang lebih akurat.
  Penerapan algoritma Random Forest akan dilakukan dengan menggunakan modul `RandomForestClassifier()` yang tersedia di library scikit-learn. Parameter n_estimators digunakan untuk menetapkan jumlah pohon (*tree*), di mana proyek ini menggunakan 100 pohon. Setelah menentukan parameter model, langkah selanjutnya adalah membangun model dan melakukan prediksi menggunakan data pengujian. Hasil dari pengujian tersebut akan dievaluasi menggunakan metrik akurasi.
  <table border="1">
  <tr>
    <th>Kelebihan Random Forest</th>
    <th>Kekurangan Random Forest</th>
  </tr>
  <tr>
    <td>Mampu menangani data dengan dimensi tinggi dan memberikan hasil yang baik pada dataset yang besar.</td>
    <td>Model dapat menjadi sulit untuk diinterpretasikan dibandingkan dengan model yang lebih sederhana seperti pohon keputusan tunggal.</td>
  </tr>
  <tr>
    <td>Menawarkan ketahanan yang tinggi terhadap overfitting, terutama ketika jumlah pohon yang digunakan cukup banyak.</td>
    <td>Waktu pelatihan dan prediksi cenderung lebih lama dibandingkan dengan algoritma yang lebih sederhana karena banyaknya pohon yang harus dihitung.</td>
  </tr>
  <tr>
    <td>Memberikan estimasi yang baik untuk feature penting dalam dataset.</td>
    <td>Penggunaan memori yang lebih besar karena harus menyimpan banyak pohon keputusan.</td>
  </tr>
  <tr>
    <td>Dapat menangani data yang hilang dan tidak memerlukan pemrosesan data yang kompleks sebelum diterapkan.</td>
    <td>Ketika digunakan untuk masalah regresi, hasilnya bisa menjadi kurang akurat jika tidak ada penyesuaian yang tepat.</td>
  </tr>
  <tr>
    <td>Fleksibel dan dapat digunakan untuk tugas klasifikasi maupun regresi.</td>
    <td>Hasil yang dihasilkan bisa bervariasi tergantung pada parameter dan set data yang digunakan, memerlukan tuning hyperparameter yang baik.</td>
  </tr>
</table>

**Kesimpulan Model:**

Gambar dibawah ini menunjukkan `confusion matrix` dari tiga model klasifikasi, yaitu K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest (RF). Matriks ini digunakan untuk mengevaluasi kinerja masing-masing model dengan membandingkan hasil prediksi terhadap nilai sebenarnya.
Model KNN menghasilkan 57 negatif benar (TN) dan 44 positif benar (TP), serta mencatat 4 positif salah (FP) dan 2 negatif salah (FN). Sementara itu, model SVM menunjukkan hasil yang cukup baik dengan 57 TN, 46 TP, dan tidak ada FN, meskipun terdapat 4 FP. Di sisi lain, model RF menunjukkan kinerja terbaik dengan 61 TN dan 46 TP, tanpa adanya FP maupun FN. Secara keseluruhan, semua model menunjukkan performa yang baik, namun **Random Forest menunjukkan akurasi tertinggi di antara ketiga model** tersebut.

<img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/conf_matrix.png" alt="Confision Matrix"/>

## Evaluation
### Perhitungan Evaluasi
Berikut adalah penjelasan mengenai metrik evaluasi yang digunakan, serta analisis hasil proyek berdasarkan 4 metrik yaitu **akurasi, precision, recall, dan F1 score**:

1. **Akurasi (Accuracy)**\
Rasio jumlah observasi yang diklasifikasikan dengan benar terhadap total observasi. Akurasi dapat menyesatkan dalam dataset yang tidak seimbang, sehingga sering dilengkapi dengan presisi dan recall. Rumusnya adalah:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?{\color{White}Akurasi=\frac{True&space;Positives&plus;True&space;Negative}{Total&space;Observasi}}">
</p>

2. **Presisi (Precision)**\
Rasio antara jumlah prediksi positif yang benar dengan total prediksi positif. Presisi penting dalam konteks di mana biaya dari false positives (prediksi positif yang salah) tinggi. Rumusnya adalah:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?{\color{White}Presisi=\frac{True&space;Positives}{True&space;Positives&plus;False&space;Negatives}}">
</p>

3. **Recall**\
Digunakan untuk mengukur kemampuan model untuk menemukan semua kasus relevan (yaitu, true positives). Recall sangat penting dalam situasi di mana kehilangan kasus positif (false negative) dapat berakibat serius. Rumusnya adalah:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?{\color{White}Recall=\frac{True&space;Positives}{True&space;Positives&plus;False&space;Negatives}}">
</p>

4. **F1 Score**\
Digunakan untuk menghitung rata-rata harmonis antara presisi dan recall, memberikan keseimbangan antara kedua metrik tersebut. Skor F1 berguna ketika Anda membutuhkan satu metrik untuk menilai kinerja, terutama pada kelas yang tidak seimbang.Rumusnya adalah:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?{\color{White}Skor&space;F1=2\ast\frac{(Presisi\ast&space;Recall)}{(Presisi&plus;Recall)}}">
</p>

Dalam hal ini, metrik evaluasi yang dipilih untuk model machine learning ini adalah **recall**, yang sangat relevan untuk deteksi penyakit. <mark>Recall akan mengukur seberapa banyak *actual positive* yang dapat diidentifikasi oleh model. Ketika seorang pasien yang menderita anemia (*actual positive*) menjalani tes dan diprediksi tidak menderita anemia (*predicted negative*), biaya yang timbul akibat *false negative* bisa sangat tinggi jika penyakit tersebut tidak segera ditangani. Ini menekankan pentingnya memiliki model dengan nilai recall yang tinggi agar tidak salah dalam memprediksi pasien yang sebenarnya menderita anemia </mark>.
dengan TP menunjukkan true positive. Nilai ideal untuk recall adalah 1, yang menunjukkan tidak adanya *false negative* (FN = 0). Ketika nilai FN meningkat, nilai penyebut akan lebih besar dibandingkan dengan pembilang, sehingga menyebabkan penurunan nilai recall.

**Keterangan:**
- **True Positive (TP)**: Model memprediksi benar dan hasil aktualnya positif.
- **True Negative (TN)**: Model memprediksi benar tetapi hasil aktualnya negatif.
- **False Positive (FP)**: Model memprediksi positif, tetapi hasil yang benar adalah negatif.
- **False Negative (FN)**: Model memprediksi negatif, tetapi hasil yang benar adalah positif.

### Penjelasan Hasil Evaluasi tiap Model
1. **K-Nearest Neighbor (KNN)**\
Berdasarkan hasil evaluasi, precision untuk kelas 0 mencapai 1.00, menunjukkan bahwa setiap prediksi kelas 0 sepenuhnya benar. Sementara itu, precision untuk kelas 1 adalah 0.92, menandakan adanya beberapa prediksi yang salah pada kelas ini. Recall model untuk kelas 0 adalah 0.93, yang berarti model mampu mendeteksi 93.4% dari semua contoh kelas 0, sedangkan recall untuk kelas 1 mencapai 1.00, yang menunjukkan bahwa model mendeteksi semua contoh kelas 1 dengan sempurna. Nilai F1-score, yang merupakan rata-rata harmonis antara precision dan recall, tercatat sebesar 0.97 untuk kelas 0 dan 0.96 untuk kelas 1, menandakan keseimbangan performa yang baik dalam mendeteksi kedua kelas. Secara keseluruhan, akurasi model tercatat sebesar 96.26%, yang mengindikasikan bahwa sebagian besar prediksi model sudah benar. Metrik macro average menunjukkan rata-rata precision, recall, dan F1-score sebesar 0.96, sementara weighted average yang mempertimbangkan distribusi data memberikan hasil serupa. Dengan demikian, model KNN ini memiliki performa yang baik dan cukup konsisten dalam mengklasifikasikan kedua kelas dengan distribusi yang seimbang.
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/eval_knn.png" />
</p>

2. **Support Vector Machine (SVM)**\
Berdasarkan hasil evaluasi, precision untuk kelas 0 mencapai 1.00, yang menunjukkan bahwa setiap prediksi untuk kelas ini benar. Precision untuk kelas 1 tercatat sebesar 0.92, menandakan adanya beberapa prediksi yang tidak tepat untuk kelas ini. Dari segi recall, model mampu mendeteksi 93.4% dari semua contoh kelas 0 dengan nilai 0.93, sedangkan recall untuk kelas 1 mencapai 1.00, yang berarti semua contoh kelas 1 terdeteksi dengan sempurna. F1-score, yang merepresentasikan keseimbangan antara precision dan recall, tercatat sebesar 0.97 untuk kelas 0 dan 0.96 untuk kelas 1, menunjukkan kinerja yang kuat di kedua kelas. Akurasi model secara keseluruhan tercatat sebesar 96.26%, yang menunjukkan bahwa sebagian besar prediksi model sudah tepat. Rata-rata metrik (macro average) precision, recall, dan F1-score di antara kedua kelas masing-masing sebesar 0.96, dan weighted average yang mempertimbangkan distribusi data menunjukkan hasil yang serupa. Secara keseluruhan, model SVM ini memiliki performa yang sangat baik dan seimbang.
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/eval_svm.png" />
</p>

3. **Random Forest (RF)**\
Hasil evaluasi model Random Forest menunjukkan performa yang sangat sempurna dalam mengklasifikasikan dua kelas (kelas 0 dan kelas 1). Precision, recall, dan F1-score untuk kedua kelas, yaitu kelas 0 dan kelas 1, semuanya tercatat sebesar 1.00. Ini berarti bahwa model tidak membuat kesalahan dalam prediksi, baik dalam hal ketepatan (precision), kemampuan mendeteksi semua contoh yang benar (recall), maupun keseimbangan antara precision dan recall (F1-score). Akurasi model juga mencapai 100%, yang menunjukkan bahwa semua prediksi model untuk data uji adalah benar. Rata-rata metrik untuk macro average dan weighted average juga bernilai 1.00, yang mengindikasikan bahwa model bekerja secara konsisten dan seimbang tanpa adanya bias terhadap kelas tertentu. Secara keseluruhan, model Random Forest ini menunjukkan performa yang sangat optimal dalam klasifikasi, dengan semua metrik evaluasi berada di tingkat maksimum.
<p align="center">
  <img src="https://raw.githubusercontent.com/rrexzra36/anemia-predictive-analytics/refs/heads/main/images/eval_rf.png" />
</p>

## Kesimpulan
- **Random Forest merupakan model paling efektif** dalam proyek ini, dengan nilai recall tertinggi di antara ketiga algoritma yang diterapkan, yaitu 1.0.
- Penelitian berhasil menjawab problem statement dengan menerapkan algoritma machine learning seperti KNN, SVM, dan Random Forest untuk memprediksi anemia menggunakan data sel darah merah. Proses pengembangan model meliputi *preprocessing data, exploratory data analysis (EDA)*, dan evaluasi kinerja model, yang berhasil menjawab tujuan prediksi anemia.
- Penelitian ini mencapai seluruh tujuan, yaitu menjelaskan penerapan algoritma machine learning, mengembangkan model prediksi anemia, dan mengevaluasi kinerjanya. Algoritma yang digunakan dievaluasi menggunakan metrik seperti ***precision***, ***recall***, ***F1-score***, dan ***accuracy***, yang menunjukkan kinerja model yang baik.
- Solusi yang diterapkan, termasuk preprocessing data, EDA, pemilihan algoritma (KNN, SVM, Random Forest), serta hyperparameter tuning, memberikan dampak signifikan terhadap performa model. Evaluasi menunjukkan bahwa model mampu memprediksi anemia dengan akurasi tinggi, mendukung deteksi dini dan pencegahan anemia secara efektif.
<hr>
<img src="https://raw.githubusercontent.com/rrexzra36/spotify-recommendation-system/refs/heads/main/images/Footer.png">