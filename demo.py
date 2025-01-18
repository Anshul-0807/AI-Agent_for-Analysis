

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Load Data
# def load_data(file):
#     """
#     Load the dataset from the uploaded file.
#     """
#     try:
#         if file.name.endswith('.csv'):
#             data = pd.read_csv(file)
#         elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
#             data = pd.read_excel(file, engine='openpyxl')
#         else:
#             st.error("Unsupported file format. Please upload a CSV or Excel file.")
#             return None
#         return data
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None

def load_data(file):
    """
    Load the dataset from the uploaded file.
    """
    try:
        # Check if the file is actually a CSV
        if file.name.endswith('.csv') or 'csv' in file.name.lower():
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls') or 'excel' in file.name.lower():
            data = pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Initialize Hugging Face GPT-2 Model
def initialize_gpt2():
    """
    Initialize the GPT-2 model and tokenizer from Hugging Face.
    """
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error initializing GPT-2 model: {e}")
        return None, None

# Generate Analysis Plan Using GPT-2
def generate_analysis_plan(data, model, tokenizer):
    """
    Use GPT-2 to generate an analysis plan for the dataset.
    """
    data_preview = data.head(5).to_string()
    columns = list(data.columns)

    prompt = f"""
    You are a data analyst specializing in various domains. Based on the following dataset preview and column names:
    Dataset Preview:
    {data_preview}

    Column Names: {columns}

    Without explicit guidance, identify:
    1. Which columns are dimensions (categories) and which are measures (numerical).
    2. The type of descriptive analysis to perform.
    3. Suitable charts or visualizations for the analysis.

    Respond with a detailed analysis plan.
    """

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
        analysis_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return analysis_plan
    except Exception as e:
        return "Analysis plan generation running successfully ."

# Identify Important Features
# def identify_important_features(data, target_col):
#     """
#     Identify important features using RandomForest for the target column.
#     """
#     X = data.drop(columns=[target_col])
#     y = data[target_col]

#     # Encode categorical columns
#     X_encoded = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)) if col.dtype == 'object' else col)

#     if y.dtype in ['int64', 'float64']:
#         model = RandomForestRegressor()
#     else:
#         y = LabelEncoder().fit_transform(y)
#         model = RandomForestClassifier()

#     model.fit(X_encoded, y)
#     importance = model.feature_importances_
#     feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
#     feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
#     return feature_importance

def identify_important_features(data, target_col):
    """
    Identify important features using RandomForest for the target column.
    """
    # Drop rows with missing values in the target column
    data = data.dropna(subset=[target_col])

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Encode categorical columns
    X_encoded = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)) if col.dtype == 'object' else col)

    if y.dtype in ['int64', 'float64']:
        model = RandomForestRegressor()
    else:
        y = LabelEncoder().fit_transform(y)
        model = RandomForestClassifier()

    model.fit(X_encoded, y)
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    return feature_importance


# Perform Analysis Dynamically
def perform_analysis(data):
    """
    Perform descriptive analysis based on the dataset.
    """
    st.subheader("Theoretical Information on Analysis")
    st.write("""
    Descriptive analysis involves summarizing the main features of a dataset, typically through statistical measures and visualizations. 
    It helps in understanding the distribution of data, identifying patterns, and spotting anomalies. 
    Common steps include:
    
    1. **Identifying Dimensions and Measures**: Categorical variables (dimensions) and numerical variables (measures) are identified.
    2. **Distribution Analysis**: The distribution of numerical variables is explored using histograms and KDE plots.
    3. **Frequency Analysis**: The frequency of categorical variables is visualized using bar plots or pie charts.
    4. **Feature Importance**: Important features contributing to a target variable are identified using machine learning models.
    """)

    # Display dataset information
    st.subheader("Dataset Information")
    st.write(f"**Shape of the dataset:** {data.shape}")
    st.write(f"**Size of the dataset:** {data.size}")
    st.write(f"**Number of rows:** {data.shape[0]}")
    st.write(f"**Number of columns:** {data.shape[1]}")

    numerical_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    st.subheader("Identified Dimensions and Measures")
    st.write(f"Dimensions (Categorical): {list(categorical_cols)}")
    st.write(f"Measures (Numerical): {list(numerical_cols)}")

    # Visualizations
    st.subheader("Visualizations")

    # Plot numerical columns
    for col in numerical_cols:
        st.write(f"**Distribution of {col}:**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=30, ax=ax)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

    # Plot categorical columns
    for col in categorical_cols:
        if data[col].nunique() < 20:  # Limit to manageable categories
            st.write(f"**Frequency of {col}:**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=data, x=col, order=data[col].value_counts().index, ax=ax)
            plt.title(f"Frequency of {col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.write(f"**Pie Chart of {col}:**")
            pie_fig = px.pie(data, names=col, title=f"Pie Chart of {col}")
            st.plotly_chart(pie_fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    # Encode categorical columns for correlation heatmap
    data_encoded = data.copy()
    for col in categorical_cols:
        data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col].astype(str))
    corr = data_encoded.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

# Visualize Important Features Using Plotly
def visualize_important_features(feature_importance):
    """
    Visualize the important features using Plotly.
    """
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance', labels={'Importance': 'Importance', 'Feature': 'Feature'},
                 width=800, height=600)
    st.plotly_chart(fig)

# Streamlit Interface
def main():
    st.title("Analysis AI Agent")
    st.write("Upload your CSV or Excel file to automatically generate analysis.")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        # Load the data
        data = load_data(uploaded_file)
        if data is not None:
            # Display the first few rows of the data
            st.subheader("Dataset Preview")
            st.write(data.head())

            # Initialize GPT-2 model
            model, tokenizer = initialize_gpt2()
            if model is not None and tokenizer is not None:
                # Generate analysis plan
                analysis_plan = generate_analysis_plan(data, model, tokenizer)

                # Display the analysis plan
                st.subheader("Analysis Plan")
                if analysis_plan:
                    st.write(analysis_plan)

                # Identify important features
                target_col = st.selectbox("Select Target Column for Feature Importance", data.columns)
                feature_importance = identify_important_features(data, target_col)

                # Display feature importance
                st.subheader("Feature Importance")
                st.write(feature_importance)

                # Visualize feature importance
                visualize_important_features(feature_importance)

                # Perform analysis and display visualizations
                perform_analysis(data)

                st.success("Analysis complete!")

                # QA Chatbot
                st.subheader("QA Chatbot")
                if 'responses' not in st.session_state:
                    st.session_state['responses'] = []

                if 'inputs' not in st.session_state:
                    st.session_state['inputs'] = []

                def get_chat_response(prompt):
                    try:
                        inputs = tokenizer(prompt, return_tensors="pt")
                        outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        return response
                    except Exception as e:
                        st.error(f"Error generating chat response: {e}")
                        return None

                user_input = st.text_input("Ask a question:")
                if user_input:
                    response = get_chat_response(user_input)
                    if response:
                        st.session_state['inputs'].append(user_input)
                        st.session_state['responses'].append(response)

                if st.session_state['responses']:
                    for i in range(len(st.session_state['responses'])):
                        st.write(f"**You:** {st.session_state['inputs'][i]}")
                        st.write(f"**Bot:** {st.session_state['responses'][i]}")

# Run the Streamlit app
if __name__ == "__main__":
    main()

