# IMDb random forest

This folder contains a small script to train a Random Forest classifier on `imdb_merged_results.csv`.

How it works
- The script creates a binary target `high_rating` where rating >= 7.0.
- It uses TF-IDF on the `storyline` text and one-hot encodes categorical columns (genres, country, platform). Numeric fields include `review_count` and parsed `year`.

Run

1. (Optional) Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the script from this directory:

```bash
python imdb_random_forest.py
```

Output
- Trained model saved as `imdb_rf_model.joblib` in the same directory.

Notes
- Assumption: binary target high_rating (rating >= 7). If you'd rather predict the numeric rating (regression) or another column, tell me and I can adjust.
