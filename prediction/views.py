from django.shortcuts import render
import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
import numpy as np
from tensorflow.keras.models import load_model
import json
import pandas as pd
from reportlab.lib.pagesizes import A4
import plotly.express as px
from io import BytesIO
from reportlab.pdfgen import canvas

def home(request):
    return render(request, 'home.html')

# Load the LSTM model (assuming the file is in the same directory as views.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
lstm_model_path = os.path.join(current_dir, 'lstm_model.h5')
lstm_model = load_model(lstm_model_path)

def preprocess_input(data):
    try:
        poverty_rate = data.get('poverty_rate')
        working_poverty_rate = data.get('working_poverty_rate')
        
        if poverty_rate is None or working_poverty_rate is None:
            return None

        poverty_rate = float(poverty_rate)
        working_poverty_rate = float(working_poverty_rate)
        
        # Normalize input data
        poverty_rate_normalized = poverty_rate / 100.0
        working_poverty_rate_normalized = working_poverty_rate / 100.0
        
        # Return as 2D array with shape (1, 2)
        normalized_data = np.array([[poverty_rate_normalized, working_poverty_rate_normalized]])
        print("Raw input data shape after preprocessing:", normalized_data.shape)  # Should be (1, 2)
        return normalized_data

    except Exception as e:
        print("Error in preprocessing:", e)
        return None

def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    corrected_prediction = prediction[0][0] * 0.9
    return float(corrected_prediction)

@csrf_exempt
def predict_lstm(request):
    if request.method == 'POST':
        try:
            # Parse JSON data
            data = json.loads(request.body)

            # Preprocess input data
            input_data = preprocess_input(data)
            if input_data is None:
                return JsonResponse({'error': 'Missing or invalid input fields'}, status=400)

            # Reshape for LSTM model input (1, 5, 2)
            lstm_input = np.pad(input_data, ((0, 4), (0, 0)), mode='constant').reshape(1, 5, 2)
            print("Prepared LSTM input shape:", lstm_input.shape)

            # Make prediction
            prediction = make_prediction(lstm_model, lstm_input)
            return JsonResponse({'predicted_suicide_rate': f"{prediction:.2f}"})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except ValueError as e:
            print("ValueError encountered:", e)
            return JsonResponse({'error': 'Invalid input or shape mismatch.'}, status=400)

    # Render the HTML form on a GET request
    return render(request, 'lstm_predict_form.html')



# Example views to handle specific model predictions
# @csrf_exempt
# def predict_cnn(request):
#     if request.method == 'POST':
#         return predict(request, 'cnn')z
#     else:
#         return render(request, 'cnn_predict_form.html')

# @csrf_exempt
# def predict_dense(request):
#     if request.method == 'POST':
#         return predict(request, 'dense')
#     else:
#         return render(request, 'dense_predict_form.html')

# @csrf_exempt
# def predict_lstm(request):
#     if request.method == 'POST':
#         return predict(request, 'lstm')
#     else:
#         return render(request, 'lstm_predict_form.html')

cnn_model_path = os.path.join(current_dir, 'cnn_model.h5')
dense_model_path = os.path.join(current_dir, 'fully_connected_model.h5')
cnn_model = load_model(cnn_model_path)
dense_model = load_model(dense_model_path)

@csrf_exempt
def compare_predictions(request):
    if request.method == 'POST':
        try:
            # Extract form data
            poverty_rate = float(request.POST.get('poverty_rate'))
            working_poverty_rate = float(request.POST.get('working_poverty_rate'))
            data = {"poverty_rate": poverty_rate, "working_poverty_rate": working_poverty_rate}

            # Use preprocess_input to scale the input consistently
            input_data = preprocess_input(data)
            if input_data is None:
                return JsonResponse({'error': 'Invalid input fields: preprocessing failed'}, status=400)

            # CNN input: Shape (1, 221, 1)
            cnn_input = np.pad(input_data, ((0, 0), (0, 219)), mode='constant').reshape(1, 221, 1)
            print("Prepared CNN input shape:", cnn_input.shape)

            # Dense input: Shape (1, 221)
            dense_input = np.pad(input_data, ((0, 0), (0, 219)), mode='constant').reshape(1, 221)
            print("Prepared Dense input shape:", dense_input.shape)

            # LSTM input: Shape (1, 5, 2)
            lstm_input = np.pad(input_data, ((0, 4), (0, 0)), mode='constant').reshape(1, 5, 2)
            print("Prepared LSTM input shape:", lstm_input.shape)

            # Get predictions
            cnn_prediction = cnn_model.predict(cnn_input)[0][0]
            dense_prediction = dense_model.predict(dense_input)[0][0]
            lstm_prediction = lstm_model.predict(lstm_input)[0][0]

            # Interpret results
            result_explanations = {
                'cnn_prediction': f"{cnn_prediction:.2f}",
                'dense_prediction': f"{dense_prediction:.2f}",
                'lstm_prediction': f"{lstm_prediction:.2f}",
                'explanation': [
                    "CNN Prediction: CNNs often identify local patterns and relationships. Lower predictions might reflect subtle correlations.",
                    "Dense Prediction: Dense models treat each feature independently, yielding generalized predictions. This score reflects the broader correlation between inputs.",
                    "LSTM Prediction: LSTMs capture sequential relationships. A high prediction here might indicate sensitivity to past data or trends."
                ]
            }

            return render(request, 'compare_results.html', {'predictions': result_explanations})

        except ValueError as e:
            print("ValueError encountered:", e)
            return JsonResponse({'error': 'Invalid input data or input shape mismatch.'}, status=400)

    return render(request, 'compare_predictions_form.html')


# Define the path to your dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'preprocessed_dataset.csv')

def visualizations(request):
    # Load the dataset
    data = pd.read_csv(csv_path)

    # Visualization 1: Poverty Rate vs. Suicide Rate
    fig_scatter = px.scatter(
        data, x='Poverty_Rate', y='Suicide_Rate_Per_100K',
        color='Country_Short_Name', hover_data=['Year', 'Sex', 'Age_Group'],
        title='Poverty Rate vs. Suicide Rate by Country'
    )
    scatter_html = fig_scatter.to_html()

    # Visualization 2: Geospatial Map of Suicide Rate
    fig_map = px.choropleth(
        data, locations='Country_Short_Name', locationmode='country names',
        color='Suicide_Rate_Per_100K', hover_name='Country_Short_Name',
        color_continuous_scale='Blues', title='Suicide Rate per 100K by Country'
    )
    map_html = fig_map.to_html()

    # Visualization 3: Trend of Suicide Rate Over Time by Country
    fig_trend = px.line(
        data, x='Year', y='Suicide_Rate_Per_100K', color='Country_Short_Name',
        title='Trend of Suicide Rate Over Time by Country'
    )
    trend_html = fig_trend.to_html()

    # Visualization 4: Box Plot for Suicide Rate by Age Group
    fig_box = px.box(
        data, x='Age_Group', y='Suicide_Rate_Per_100K', color='Age_Group',
        title='Suicide Rate Distribution by Age Group'
    )
    box_html = fig_box.to_html()

    # Visualization 5: Histogram for Suicide Rate Distribution
    fig_hist = px.histogram(
        data, x='Suicide_Rate_Per_100K', nbins=30,
        title='Distribution of Suicide Rate (Per 100K)',
        labels={'Suicide_Rate_Per_100K': 'Suicide Rate per 100K'}
    )
    hist_html = fig_hist.to_html()

    # Visualization 6: Comparison Line Chart of Poverty and Working Poverty Rate Over Time
    fig_comparison = px.line(
        data, x='Year', y=['Poverty_Rate', 'Working_Poverty_Rate'], color='Country_Short_Name',
        title='Poverty Rate vs. Working Poverty Rate Over Time by Country'
    )
    comparison_html = fig_comparison.to_html()

    # Visualization 7: Heatmap of Suicide Rate by Country and Year (fixing duplicate entries)
    # Aggregate by Year and Country_Short_Name, taking the mean for each group
    heatmap_data = data.groupby(['Year', 'Country_Short_Name'])['Suicide_Rate_Per_100K'].mean().unstack()
    fig_heatmap = px.imshow(
        heatmap_data, aspect='auto', color_continuous_scale='Viridis',
        title='Heatmap of Suicide Rate by Country and Year'
    )
    heatmap_html = fig_heatmap.to_html()

    return render(request, 'visualizations.html', {
        'scatter_html': scatter_html,
        'map_html': map_html,
        'trend_html': trend_html,
        'box_html': box_html,
        'hist_html': hist_html,
        'comparison_html': comparison_html,
        'heatmap_html': heatmap_html,
    })


def download_report(request):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="prediction_report.pdf"'

    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    # Title Page
    p.setFont("Helvetica-Bold", 24)
    p.drawCentredString(width / 2, height - 100, "Suicide Rate Prediction Report")
    p.setFont("Helvetica", 14)
    p.drawCentredString(width / 2, height - 150, "An In-Depth Analysis Using Machine Learning Models")
    p.drawString(100, height - 250, "Prepared by: Your Organization Name")
    p.drawString(100, height - 270, "Date: 2024-11-05")
    p.showPage()

    # Introduction
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height - 100, "Introduction")
    p.setFont("Helvetica", 12)
    p.drawString(100, height - 150, "This report presents a comprehensive analysis of suicide rate predictions.")
    p.drawString(100, height - 170, "The aim of this project is to provide accurate and insightful predictions")
    p.drawString(100, height - 190, "of suicide rates using various machine learning models. The key socioeconomic")
    p.drawString(100, height - 210, "factors considered in this analysis are poverty rate and working poverty rate,")
    p.drawString(100, height - 230, "which have shown significant correlations with mental health outcomes.")
    p.showPage()

    # Data Summary
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height - 100, "Data Summary")
    p.setFont("Helvetica", 12)
    p.drawString(100, height - 150, "The dataset used for this analysis includes socioeconomic data for various")
    p.drawString(100, height - 170, "regions. The primary features considered are:")
    p.drawString(120, height - 190, "- Poverty Rate")
    p.drawString(120, height - 210, "- Working Poverty Rate")
    p.drawString(100, height - 250, "These features are known to impact mental well-being and contribute to the")
    p.drawString(100, height - 270, "risk of suicide in different populations.")
    p.showPage()

    # Model Predictions
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height - 100, "Model Predictions")
    p.setFont("Helvetica", 12)
    p.drawString(100, height - 150, "The following machine learning models were used to predict suicide rates:")
    p.drawString(120, height - 170, "- Convolutional Neural Network (CNN)")
    p.drawString(120, height - 190, "- Dense Neural Network (Fully Connected Model)")
    p.drawString(120, height - 210, "- Long Short-Term Memory (LSTM) Network")
    p.drawString(100, height - 250, "Predicted Rates (Example):")
    p.drawString(120, height - 270, "CNN Prediction: 12.4 per 100,000")
    p.drawString(120, height - 290, "Dense Prediction: 13.7 per 100,000")
    p.drawString(120, height - 310, "LSTM Prediction: 11.5 per 100,000")
    p.drawString(100, height - 350, "The predictions vary based on each model's approach to feature processing.")
    p.showPage()

    # Feature Impact
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height - 100, "Feature Impact")
    p.setFont("Helvetica", 12)
    p.drawString(100, height - 150, "Poverty rate and working poverty rate are primary indicators considered by")
    p.drawString(100, height - 170, "the models. Higher values in these features tend to correlate with increased")
    p.drawString(100, height - 190, "suicide rates. The models use these features to detect patterns that may not")
    p.drawString(100, height - 210, "be obvious through traditional analysis.")
    p.drawString(100, height - 250, "For instance, regions with high poverty rates are more likely to have populations")
    p.drawString(100, height - 270, "under severe financial stress, impacting mental health.")
    p.showPage()

    # Conclusion and Recommendations
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height - 100, "Conclusion and Recommendations")
    p.setFont("Helvetica", 12)
    p.drawString(100, height - 150, "The analysis provides valuable insights into the potential impact of poverty")
    p.drawString(100, height - 170, "on suicide rates. It is recommended to continue monitoring these indicators")
    p.drawString(100, height - 190, "and update the models with new data for improved accuracy. Further exploration")
    p.drawString(100, height - 210, "of additional socioeconomic factors could enhance prediction quality.")
    p.showPage()

    p.save()
    return response


GLOBAL_AVG_SUICIDE_RATE = 9.0  # Reference global average suicide rate




@csrf_exempt
def real_time_prediction(request):
    return render(request, 'real_time_prediction.html', {'global_avg_rate': GLOBAL_AVG_SUICIDE_RATE})
