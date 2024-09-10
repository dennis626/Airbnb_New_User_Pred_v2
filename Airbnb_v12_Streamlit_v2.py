import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load EDA data 'user_data.csv'
user_data = pd.read_csv('train_users_2.csv') 
# Load 'result_Best_Score.csv'
sub_whole_df = pd.read_csv('result_Best_Score_v2.csv')

# Define a sidebar with four different pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a Page", ["1.0 Model Performance Metrics", "2.0 Visual Analysis (EDA)", "3.0 Dropdown for test_ids", "4.0 Dropdown for lbl_encoder"])

# Function to display model performance metrics
def display_model_performance_metrics():
    st.write("### Model Performance Metrics")
    results = {
        "Model": ["XGBoost", "Logistic Regression (Bayesian)"],
        "Accuracy": [0.65, 0.61],
        "Precision": [0.59, 0.54],
        "Recall": [0.65, 0.62],
        "F1-Score": [0.60, 0.59],
        "NDCG Score": [0.83, 0.82]
    }
    results_df = pd.DataFrame(results)

    st.dataframe(results_df)

    # Set up the figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Create a bar width and x positions for bars
    bar_width = 0.2
    x = np.arange(len(results_df['Model']))

    # Plot Accuracy, Recall, Precision, F1-Score (on the first subplot)
    ax[0].bar(x - bar_width*1.5, results_df['Accuracy'], width=bar_width, label="Accuracy")
    ax[0].bar(x - bar_width/2, results_df['Recall'], width=bar_width, label="Recall")
    ax[0].bar(x + bar_width/2, results_df['Precision'], width=bar_width, label="Precision")
    ax[0].bar(x + bar_width*1.5, results_df['F1-Score'], width=bar_width, label="F1-Score")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(results_df['Model'])
    ax[0].legend()
    ax[0].set_title("Accuracy, Recall, Precision, F1-Score Comparison")

    # Plot NDCG Score (on the second subplot)
    ax[1].bar(results_df['Model'], results_df['NDCG Score'], label="NDCG Score", color='purple')
    ax[1].legend()
    ax[1].set_title("NDCG Score Comparison")

    st.pyplot(fig)

# Function to display visual analysis or EDA
def display_visual_analysis():
    st.write("### Visual Analysis (EDA)")

    # Age Distribution
    st.write("#### Age Distribution (Ages 0 - 120)")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(user_data[user_data['age'].between(0, 120)]['age'], ax=ax)
    ax.set_xlim(0, 120)  # Set the x-axis limit from 0 to 120
    st.pyplot(fig)

    # Signup Flow Distribution
    st.write("#### Signup Flow Distribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(user_data['signup_flow'], ax=ax)
    st.pyplot(fig)

    # Gender Distribution
    st.write("#### Gender Distribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    counts = user_data['gender'].fillna('NaN').value_counts(dropna=False)
    sns.countplot(x=user_data['gender'].fillna('NaN'), order=counts.index, ax=ax)
    for i in range(len(counts)):
        ax.text(i, counts[i] + 1200, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=10)
    st.pyplot(fig)

    # Gender-Age Distribution
    st.write("#### Gender-Age Distribution")
    filtered_df = user_data[(user_data['age'] >= 18) & (user_data['age'] <= 100)]
    fig = px.box(filtered_df, x='gender', y='age', color='gender', title='Gender-Age Distribution', labels={'age': 'Age', 'gender': 'Gender'})
    st.plotly_chart(fig)

    # Signup Method Distribution
    st.write("#### Signup Method Distribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(x=user_data['signup_method'], ax=ax)
    for i in range(len(counts)):
        ax.text(i, counts[i] + 1200, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=10)
    st.pyplot(fig)

    # Affiliate Provider Distribution
    st.write("#### Affiliate Provider Distribution")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(y=user_data['affiliate_provider'], ax=ax)
    st.pyplot(fig)

    # Affiliate Channel Distribution
    st.write("#### Affiliate Channel Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y=user_data['affiliate_channel'], ax=ax)
    st.pyplot(fig)

    # Affiliate Channel Flows Treemap
    st.write("#### Affiliate Channel Flows Distribution")
    signup_flow_dist = user_data['affiliate_channel'].value_counts().reset_index()
    signup_flow_dist.columns = ['affiliate_channel', 'count']
    fig = px.treemap(signup_flow_dist, path=['affiliate_channel'], values='count', title='Affiliate Channel Flows Distribution')
    st.plotly_chart(fig)

    # Language Distribution
    st.write("#### Language Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x=user_data['language'], ax=ax)
    st.pyplot(fig)

    # First Device Type Distribution
    st.write("#### First Device Type Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y=user_data['first_device_type'], ax=ax)
    st.pyplot(fig)

    # Booking Date Month Distribution
    st.write("#### Booking Date Month Distribution")
    months_freq = user_data['date_first_booking'].dropna().dt.month_name().str[:3]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=months_freq, order=months_freq.value_counts().index, ax=ax)
    st.pyplot(fig)

    # Booking Date Week Day Distribution
    st.write("#### Booking Date Week Day Distribution")
    week_days_freq = user_data['date_first_booking'].dropna().dt.day_name()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=week_days_freq, order=week_days_freq.value_counts().index, ax=ax)
    st.pyplot(fig)

    # Destination Country Distribution
    st.write("#### Destination Country Distribution")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=user_data['country_destination'], ax=ax)
    st.pyplot(fig)

    # Account Creation Date Frequency
    st.write("#### Account Creation Date Frequency")
    creation_dates = user_data['date_account_created'].value_counts()
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.lineplot(x=creation_dates.index, y=creation_dates.values, ax=ax)
    st.pyplot(fig)

    # Monthly Trends in Account Creation
    st.write("#### Monthly Trends in Account Creation")
    user_data['date_account_created'] = pd.to_datetime(user_data['date_account_created'])
    user_data['year_month'] = user_data['date_account_created'].dt.to_period('M').astype(str)
    monthly_counts = user_data.groupby('year_month').size().reset_index(name='counts')
    fig = px.line(monthly_counts, x='year_month', y='counts', title='Monthly Trends in Account Creation')
    st.plotly_chart(fig)

    # Monthly Trends in Account Creation for Users Traveling to the US
    st.write("#### Monthly Trends in Account Creation for Users Traveling to the US")
    us_travel_data = user_data[user_data['country_destination'] == 'US']
    monthly_counts = us_travel_data.groupby('year_month').size().reset_index(name='counts')
    fig = px.line(monthly_counts, x='year_month', y='counts', title='Monthly Trends for US Travelers')
    st.plotly_chart(fig)

    # Destination Country Distribution Per Gender
    st.write("#### Destination Country Distribution Per Gender")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.countplot(x='country_destination', hue='gender', data=user_data, ax=ax)
    st.pyplot(fig)

    # Destination Country Distribution Per Age
    st.write("#### Destination Country Distribution Per Age")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='country_destination', y='age', data=user_data, ax=ax)
    st.pyplot(fig)

    # Signup Method vs. Destination Country (Stacked Bar)
    st.write("#### Signup Method vs. Destination Country")
    signup_vs_country = user_data.groupby(['signup_method', 'country_destination']).size().reset_index(name='count')
    fig = px.bar(signup_vs_country, x='signup_method', y='count', color='country_destination', title='Signup Method vs. Destination Country', barmode='stack')
    st.plotly_chart(fig)

    # Parallel Categories Plot of Gender, Signup Method, and Destination Country
    st.write("#### Parallel Categories Plot of Gender, Signup Method, and Destination Country")
    fig = px.parallel_categories(user_data, dimensions=['gender', 'signup_method', 'country_destination'], title='Parallel Categories Plot')
    st.plotly_chart(fig)

    # Cohort Analysis by Destination Country
    st.write("#### Cohort Analysis by Destination Country")
    user_data['cohort_month'] = user_data['date_account_created'].dt.to_period('M')
    cohort_counts = user_data.groupby(['cohort_month', 'country_destination']).size().unstack().fillna(0)
    cohort_percent = cohort_counts.divide(cohort_counts.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for year in sorted(user_data['cohort_month'].dt.year.unique()):
        filtered_data = cohort_percent[cohort_percent.index.year == year]
        fig.add_trace(go.Heatmap(z=filtered_data.T.values, x=filtered_data.index, y=filtered_data.columns, colorscale='Blues', visible=False, name=str(year)))
    fig.data[0].visible = True
    st.plotly_chart(fig)

    # Whether Members Booked Per Gender
    st.write("#### Whether Members Booked Per Gender")
    booked_status = user_data['date_first_booking'].notna()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=booked_status, hue='gender', data=user_data, ax=ax)
    st.pyplot(fig)

    # Monthly Booking Trends
    st.write("#### Monthly Booking Trends")
    user_data['year_month_booking'] = user_data['date_first_booking'].dt.to_period('M').astype(str)
    monthly_bookings = user_data.groupby('year_month_booking').size().reset_index(name='counts')
    fig = px.line(monthly_bookings, x='year_month_booking', y='counts', title='Monthly Booking Trends')
    st.plotly_chart(fig)

    # Target Variable Distribution Per Country Destination (Pie Chart)
    st.write("#### Target Variable Distribution Per Country Destination")
    country_dist = user_data['country_destination'].value_counts().reset_index()
    country_dist.columns = ['country_destination', 'count']
    fig = px.pie(country_dist, values='count', names='country_destination', title='Target Variable Distribution Per Country Destination')
    st.plotly_chart(fig)

    # Signup Method and Device Type Hierarchical Distribution (Sunburst Plot)
    st.write("#### Signup Method and Device Type Hierarchical Distribution")
    user_data['count'] = 1
    fig = px.sunburst(user_data, path=['signup_method', 'first_device_type'], values='count', title='Signup Method and Device Type Hierarchical Distribution')
    st.plotly_chart(fig)

    # Geographical Distribution of Destinations Country (Choropleth Map)
    st.write("#### Geographical Distribution of Destinations Country")
    country_iso_mapping = {
        'US': 'USA',
        'FR': 'FRA',
        'CA': 'CAN',
        'GB': 'GBR',
        'ES': 'ESP',
        'IT': 'ITA',
        'PT': 'PRT',
        'NL': 'NLD',
        'DE': 'DEU',
        'AU': 'AUS',
        'other': 'Other'
    }
    country_dist['country_destination'] = country_dist['country_destination'].map(country_iso_mapping)
    fig = px.choropleth(country_dist, locations='country_destination', locationmode='ISO-3', color='count', title='Geographical Distribution of Destinations Country', color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig)

# Function for dropdown to filter test_ids
def dropdown_test_ids():
    selected_test_id = st.selectbox("Select User ID", sub_whole_df['id'].unique())
    filtered_by_test_id = sub_whole_df[sub_whole_df['id'] == selected_test_id]
    st.write(f"### Country results for ID: {selected_test_id}")
    st.write(filtered_by_test_id)

    # Add the remark message
    st.markdown("""
    **Country Notes:**
    - NDF: No Destination Found
    - US: United States
    - FR: France
    - IT: Italy
    - GB: Great Britain/United Kingdom
    - ES: Spain
    - CA: Canada
    - DE: Germany
    - NL: Netherlands
    - AU: Australia
    - PT: Portugal
    """)

# Function for dropdown to filter lbl_encoder
def dropdown_lbl_encoder():
    selected_lbl_encoder = st.selectbox("Select Country", sub_whole_df['country'].unique())
    filtered_by_lbl_encoder = sub_whole_df[sub_whole_df['country'] == selected_lbl_encoder]
    st.write(f"### User ID for Country {selected_lbl_encoder}")
    st.write(filtered_by_lbl_encoder)

    # Add the remark message
    st.markdown("""
    **Country Notes:**
    - NDF: No Destination Found
    - US: United States
    - FR: France
    - IT: Italy
    - GB: Great Britain/United Kingdom
    - ES: Spain
    - CA: Canada
    - DE: Germany
    - NL: Netherlands
    - AU: Australia
    - PT: Portugal
    """)

# Page navigation logic
if page == "1.0 Model Performance Metrics":
    display_model_performance_metrics()
elif page == "2.0 Visual Analysis (EDA)":
    display_visual_analysis()
elif page == "3.0 Dropdown for test_ids":
    dropdown_test_ids()
elif page == "4.0 Dropdown for lbl_encoder":
    dropdown_lbl_encoder()
