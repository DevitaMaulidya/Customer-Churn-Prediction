from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UploadCSVForm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os
import subprocess
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def handle_uploaded_file(f):
    upload_path = os.path.join(settings.MEDIA_ROOT, f.name)
    with open(upload_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return upload_path

def run_notebooks(file_path):
    notebook_paths = [
        os.path.join(settings.BASE_DIR, 'notebooks', 'generate_img.ipynb'),
    ]

    # Jalankan setiap notebook
    for i, notebook_path in enumerate(notebook_paths):
        command = [
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            '--ExecutePreprocessor.timeout=600',  # Timeout jika notebook membutuhkan waktu lama
            '--output',
            f'output_{i+1}.ipynb',
            notebook_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error menjalankan notebook {i+1}:", result.stderr)
        else:
            print(f"Notebook {i+1} berhasil dijalankan")
   
def upload_csv_view(request):
    BASE_DIR = settings.BASE_DIR
    data_path = os.path.join(BASE_DIR, 'data', 'Customer-Churn-Records.csv')
    df = pd.read_csv(data_path)
    
    result = None
    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_path = handle_uploaded_file(request.FILES['csv_file'])
            df = pd.read_csv(csv_path)

            # Validasi kolom
            expected_columns = ['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Complain', 'SatisfactionScore', 'CardType', 'PointEarned']
            if list(df.columns) != expected_columns:
                result = 'Kolom dalam file tidak sesuai!'
            else:
                run_notebooks(csv_path)  # Hanya dipanggil saat POST valid
                return redirect('dashboard')
    else:
        form = UploadCSVForm()
        
    return render(
        request, 
        'upload.html', 
        {'form': form, 
         'result': result, 
         "data": df.head(5).to_dict(orient='records')
         },
    )

def dashboard(request):
    ### DATA INPUT
    media_folder = os.path.join(settings.BASE_DIR, 'media')
    csv_files = [f for f in os.listdir(media_folder) if f.endswith('.csv')]
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(media_folder, x)))
    df = pd.read_csv(os.path.join(media_folder, latest_file))
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
    
    #### MODEL
    model_pathd = os.path.join(settings.BASE_DIR, 'churnapp', 'churn_model_default.pkl')
    model_patho = os.path.join(settings.BASE_DIR, 'churnapp', 'churn_model_optimasi.pkl')
    
    row_count = len(df)
    avg_credit_score = round(df['CreditScore'].mean())
    avg_balance = round(df['Balance'].mean())
    
    display_datad = df[['Surname']].copy()
    display_datao = df[['Surname']].copy()
    
    # Siapkan fitur untuk prediksi
    X_test = df.drop(['CustomerId', 'Surname', 'Complain'], axis = 1)
    
    with open(model_pathd, 'rb') as f:
        model = pickle.load(f)
        
    with open(model_patho, 'rb') as f:
        model1 = pickle.load(f)
    
    df["Geography"] = df["Geography"]
    display_datad["Geography"] = df["Geography"]
    display_datao["Geography"] = df["Geography"]
    
    df["Gender"] = df["Gender"]
    display_datad["Gender"] = df["Gender"]
    display_datao["Gender"] = df["Gender"]
    
    y_predd = model.predict(X_test)
    y_predo = model1.predict(X_test)
    
    label_map = {0: 'No Churn', 1: 'Churn'}
    display_datad['Prediction'] = [label_map.get(pred, pred) for pred in y_predd]
    display_datao['Prediction'] = [label_map.get(pred, pred) for pred in y_predo]    
    
    churn_rate = (y_predd.sum() / len(y_predd)) * 100

    return render(
        request,
        "dashboard.html",
        {
            "churn_rate": round(churn_rate),
            "dataCount": row_count,
            'avg_credit_score': avg_credit_score,
            'avg_balance': avg_balance,
            "data": df.head(1000).to_dict(orient='records'),
            "filename": latest_file, 
            'data_selected': display_datad.to_dict(orient='records'),
            'data_selected1': display_datao.to_dict(orient='records'),
        },
    )
