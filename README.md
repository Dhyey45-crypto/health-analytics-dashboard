# 🏥 Health Analytics Dashboard
### 🔗 [Launch the Live App](https://health-analytics-dashboard-d2rj64a9axqekzjdvbwyak.streamlit.app/)

An interactive **Health Analytics Dashboard** built with **Streamlit**, **Plotly**, and **TextBlob**.
Upload a patient dataset (or use the bundled sample data) and instantly explore KPIs, age
distribution, disease distribution, BMI analysis, blood pressure analysis, gender analysis,
and patient-feedback sentiment.

---

## 📁 Project Structure

```
health_analytics_dashboard/
│
├── app.py                        # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── sample_data/
│   └── patients_sample.csv        # 200-row synthetic sample dataset
│
├── utils/
│   ├── __init__.py
│   └── data_processor.py          # Data cleaning, validation & enrichment helpers
│
└── gen_sample.py                  # Script used to (re)generate the sample dataset
```

---

## ✅ Prerequisites

- **Python 3.9+** installed
- **VS Code** installed with the **Python extension** (recommended)
- Internet connection (for the first-time install of packages + TextBlob corpora)

---

## 🚀 Step-by-Step: Run in VS Code

### 1. Unzip and open the project
Unzip the downloaded file, then in VS Code:
`File → Open Folder... → select "health_analytics_dashboard"`

### 2. Open a terminal in VS Code
`Terminal → New Terminal` (make sure it opens in the project root folder)

### 3. (Recommended) Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```
VS Code may prompt "Select Python Interpreter" — choose the one inside `venv`.

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the TextBlob corpora (one-time step, required for sentiment analysis)
```bash
python -m textblob.download_corpora
```

### 6. Run the Streamlit app
```bash
streamlit run app.py
```

### 7. View the dashboard
Streamlit will automatically open your browser. If it doesn't, go to:
```
http://localhost:8501
```

To stop the app, go back to the terminal and press `CTRL + C`.

---

## 📊 Using the Dashboard

1. **Upload Data**: In the left sidebar, upload your own patient CSV, or leave
   **"Use sample dataset"** checked to explore with the bundled synthetic data (200 patients).
2. **Filter**: Use the sidebar filters (Gender, Disease, Age range) to narrow down the data —
   all charts and KPIs update automatically.
3. **Explore Tabs**:
   - 👥 **Age Distribution** — histogram + age-group breakdown
   - 🦠 **Disease Distribution** — bar chart, pie chart, and an age-vs-disease heatmap
   - ⚖️ **BMI Analysis** — BMI histogram by category, category share, BMI vs Age scatter
   - ❤️ **Blood Pressure** — systolic vs diastolic scatter, category share, BP by age group
   - 🚻 **Gender Analysis** — gender split, BMI by gender box plot, disease by gender
   - 💬 **Feedback Sentiment** — TextBlob-powered sentiment on the `Feedback` column
4. **Download**: Expand "View Filtered Raw Data" at the bottom to preview and download the
   currently filtered dataset as CSV.

---

## 📋 Expected CSV Format

Your uploaded CSV must contain at least these columns (column names are case-sensitive):

| Column        | Type    | Example        |
|---------------|---------|----------------|
| PatientID     | number  | 1              |
| Name          | text    | Patient_1      |
| Age           | number  | 45             |
| Gender        | text    | Male / Female / Other |
| Disease       | text    | Diabetes       |
| BMI           | number  | 27.4           |
| BP_Systolic   | number  | 128            |
| BP_Diastolic  | number  | 82             |

**Optional columns** (enable extra features):
- `Feedback` (text) — enables the Sentiment Analysis tab (TextBlob)
- `AdmissionDate` (date) — reserved for future time-based analysis

A ready-to-use example is in `sample_data/patients_sample.csv`.

---

## 🛠️ Tech Stack

- **Streamlit** — dashboard UI framework
- **Plotly Express / Graph Objects** — interactive charts
- **Pandas / NumPy** — data wrangling
- **TextBlob** — sentiment analysis (polarity) on patient feedback text

---

## 🔧 Troubleshooting

- **`streamlit: command not found`** → Make sure your virtual environment is activated,
  then re-run `pip install -r requirements.txt`.
- **TextBlob sentiment errors / `LookupError`** → Run
  `python -m textblob.download_corpora` again.
- **Port already in use** → Run `streamlit run app.py --server.port 8502` and open
  `http://localhost:8502`.
- **Uploaded CSV rejected** → Check that column names exactly match the required list above.

---

## 📄 License
Free to use and modify for learning, coursework, and internal analytics use cases.
