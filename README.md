# NLP Project Data Notları

Bu dosya sadece proje dizinindeki data klasörlerinin ne işe yaradığını açıklamak için yazıldı.

## Genel klasör yapısı

```text
nlpproject/
├── admire2_training/
├── admire2_submission/
└── README.md
```

## admire2_training

Model geliştirme, eğitim ve labelled değerlendirme için kullanılan data klasörü.

```text
admire2_training/
├── EN/
└── PT/
```

### EN

İngilizce data klasörü.

```text
EN/
├── train/
├── dev/
├── test/
└── xeval/
```

* `train`: Eğitim datası.
* `dev`: Geliştirme / model kontrol datası.
* `test`: Test datası.
* `xeval`: Extended evaluation datası.

### PT

Portekizce data klasörü.

PT tarafında da train/dev/test/evaluation dosyaları bulunabilir.

## admire2_submission

Final prediction / submission üretmek için kullanılan data klasörü.

```text
admire2_submission/
├── admire2_sentences/
└── admire2_data/
```

### admire2_sentences

Farklı diller için submission cümle dosyalarını içerir.

```text
admire2_sentences/
├── submission_Chinese.tsv
├── submission_Georgian.tsv
├── submission_Greek.tsv
├── submission_Igbo.tsv
├── submission_Kazakh.tsv
├── submission_Norwegian.tsv
├── submission_Portuguese-Brazil.tsv
├── submission_Portuguese-Portugal.tsv
├── submission_Russian.tsv
├── submission_Serbian.tsv
├── submission_Slovak.tsv
├── submission_Slovenian.tsv
├── submission_Spanish-Ecuador.tsv
├── submission_Turkish.tsv
└── submission_Uzbek.tsv
```

Bu dosyalar farklı dillerdeki cümleleri ve hedef ifadeleri içerir.

### admire2_data

Submission tarafında kullanılacak image klasörlerini içerir.

```text
admire2_data/
├── Chinese/
├── Georgian/
├── Greek/
├── Igbo/
├── Kazakh/
├── Norwegian/
├── Portuguese-Brazil/
├── Portuguese-Portugal/
├── Russian/
├── Serbian/
├── Slovak/
├── Slovenian/
├── Spanish-Ecuador/
├── Turkish/
└── Uzbek/
```

Her dil klasörünün içinde o dile ait compound/deyim klasörleri ve image dosyaları bulunur.

Örnek:

```text
admire2_data/<Dil>/
├── <deyim_1>/
├── <deyim_2>/
├── <deyim_3>/
├── <deyim_4>/
└── ...
```

Her `<deyim>/` klasörü o deyime ait görüntü dosyalarını içerir.
