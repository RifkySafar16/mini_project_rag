QUESTIONS = [
    {
        "id": 1,
        "type": "direct",
        "question": "Berapa denda kalau saya telat mengembalikan buku?",
        "expected_section": "4. Denda Keterlambatan",
    },
    {
        "id": 2,
        "type": "paraphrased",
        "question": "Sampai jam berapa saya bisa baca di perpustakaan?",
        "expected_section": "1. Jam Operasional",
    },
    {
        "id": 3,
        "type": "other_section",
        "question": "Bagaimana cara memesan ruang diskusi?",
        "expected_section": "5. Fasilitas",
    },
    {
        "id": 4,
        "type": "unsupported",
        "question": "Apakah perpustakaan menyediakan layanan antar buku ke rumah?",
        "expected_section": None,   # tidak ada di dataset
    },
]