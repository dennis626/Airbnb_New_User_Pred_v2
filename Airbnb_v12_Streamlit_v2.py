import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# Add Airbnb logo to the sidebar
image = Image.open('Airbnb_logo.jpg')
st.sidebar.image(image, use_column_width=True)

# Title with pattern and custom styles
st.markdown('<h1 style="text-align:center; color:#2C3E50;">Airbnb New User Booking Prediction</h1>', unsafe_allow_html=True)
st.markdown('--------------------------------')

# Load EDA data 'user_data.csv'
user_data = pd.read_csv('train_users_2.csv') 
# Load 'result_Best_Score.csv'
sub_whole_df = pd.read_csv('result_Best_Score_v2.csv')

# Define a sidebar with four different pages
st.sidebar.markdown('<h3 style="color:#3498DB;">Navigation</h3>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Select a Page", ["1.0 Model Performance Metrics", "2.0 Visual Analysis (EDA)", "3.0 Dropdown for User ID", "4.0 Dropdown for Country"])

# Function to display model performance metrics
def display_model_performance_metrics():
    st.write("### Model Performance Metrics")

    # Using expander to add more details about the metrics
    with st.expander("Click to see the description of the metrics"):
        st.write("""
            - **Accuracy**: The percentage of correct predictions.
            - **Precision**: The proportion of positive identifications that are correct.
            - **Recall**: The proportion of actual positives that were identified correctly.
            - **F1-Score**: The harmonic mean of precision and recall.
            - **NDCG Score**: Normalized Discounted Cumulative Gain, used for ranking quality.
        """)
    
    results = {
        "Model": ["XGBoost", "Logistic Regression (Bayesian)"],
        "Accuracy": [0.65, 0.62],
        "Precision": [0.58, 0.54],
        "Recall": [0.65, 0.62],
        "F1-Score": [0.60, 0.56],
        "NDCG Score": [0.83, 0.82]
    }
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Set up the figure and axes for comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
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
    plt.xlabel('Age')
    plt.title('Age Distribution')
    sns.despine()
    st.pyplot(fig)

    # Signup Flow Distribution
    st.write("#### Signup Flow Distribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(user_data['signup_flow'], ax=ax)
    plt.xlabel('Signup Flow')
    plt.title('Signup Flow Distribution')
    sns.despine()
    st.pyplot(fig)

    # Gender Distribution
    st.write("#### Gender Distribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    counts = user_data['gender'].fillna('NaN').value_counts(dropna=False)
    sns.countplot(x=user_data['gender'].fillna('NaN'), order=counts.index, ax=ax)
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Gender Distribution')
    for i in range(counts.shape[0]):
        plt.text(i, counts[i]+1200, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=10)
    sns.despine()
    st.pyplot(fig)

    # Gender-Age Distribution
    st.write("#### Gender-Age Distribution")
    filtered_df = user_data[(user_data['age'] >= 18) & (user_data['age'] <= 100)]
    fig = px.box(filtered_df, x='gender', y='age', color='gender', title='Gender-Age Distribution', labels={'age': 'Age', 'gender': 'Gender'})
    st.plotly_chart(fig)

    # Signup Method Distribution
    st.write("#### Signup Method Distribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    counts = user_data['signup_method'].fillna('NaN').value_counts(dropna=False)
    sns.countplot(x=user_data['signup_method'].fillna('NaN'), order=counts.index, ax=ax)
    plt.xlabel('Signup Method')
    plt.ylabel('Count')
    plt.title('Signup Method Distribution')
    for i in range(counts.shape[0]):
        plt.text(i, counts[i]+1200, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=10)
    sns.despine()
    st.pyplot(fig)

    # Affiliate Provider Distribution
    st.write("#### Affiliate Provider Distribution")
    fig, ax = plt.subplots(figsize=(10, 7))
    counts = user_data['affiliate_provider'].fillna('NaN').value_counts(dropna=False)
    sns.countplot(y=user_data['affiliate_provider'].fillna('NaN'), order=counts.index, ax=ax)
    plt.ylabel('Affiliate Provider')
    plt.xlabel('Count')
    plt.title('Affiliate Provider Distribution')
    for i in range(counts.shape[0]):
        plt.text(counts[i]+5200, i+0.17, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=9)
    sns.despine()
    st.pyplot(fig)

    # Affiliate Channel Distribution
    st.write("#### Affiliate Channel Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = user_data['affiliate_channel'].fillna('NaN').value_counts(dropna=False)
    sns.countplot(y=user_data['affiliate_channel'].fillna('NaN'), order=counts.index, ax=ax)
    plt.ylabel('Affiliate Channel')
    plt.xlabel('Count')
    plt.title('Affiliate Channel Distribution')
    for i in range(counts.shape[0]):
        plt.text(counts[i]+5200, i+0.09, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=10)
    sns.despine()
    st.pyplot(fig)

    # Affiliate Channel Flows Treemap
    st.write("#### Affiliate Channel Flows Distribution")
    signup_flow_dist = user_data['affiliate_channel'].value_counts().reset_index()
    signup_flow_dist.columns = ['affiliate_channel', 'count']
    fig = px.treemap(signup_flow_dist, path=['affiliate_channel'], values='count', title='Affiliate Channel Flows Distribution')
    sns.despine()    
    st.plotly_chart(fig)

    # Language Distribution
    st.write("#### Language Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = user_data['language'].fillna('NaN').value_counts(dropna=False)
    sns.countplot(x=user_data['language'], ax=ax)
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.title('Language Distribution')
    for i in range(counts.shape[0]):
        plt.text(i, counts[i]+1000, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=6)
    sns.despine()
    st.pyplot(fig)

    # First Device Type Distribution
    st.write("#### First Device Type Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = user_data['first_device_type'].fillna('NaN').value_counts(dropna=False)
    sns.countplot(y=user_data['first_device_type'].fillna('NaN'), order=counts.index, ax=ax)
    plt.ylabel('First Device Type')
    plt.xlabel('Count')
    plt.title('First Device Type Distribution')
    for i in range(counts.shape[0]):
        plt.text(counts[i]+4000, i+0.09, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=10)
    sns.despine()
    st.pyplot(fig)

    # Ensure 'date_first_booking' is in datetime format
    user_data['date_first_booking'] = pd.to_datetime(user_data['date_first_booking'], errors='coerce')
    
    # Booking Date Month Distribution
    st.write("#### Booking Date Month Distribution")
    months_freq = user_data['date_first_booking'].dropna().dt.month_name().str[:3]
    counts = months_freq.value_counts()
    counts_order = counts.index
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=months_freq, order=counts_order, ax=ax)
    ax.set_xlabel('Booking Date Month')
    ax.set_ylabel('Count')
    ax.set_title('Booking Date Month Distribution')
    for i in range(counts.shape[0]):
        ax.text(i, counts[i] + 100, f"{counts[i] / months_freq.shape[0] * 100:.2f}%", ha='center', fontsize=9)
    sns.despine()
    st.pyplot(fig)

    # Booking Date Week Day Distribution
    st.write("#### Booking Date Week Day Distribution")
    week_days_freq = user_data['date_first_booking'].dropna().dt.day_name()
    counts = week_days_freq.value_counts()
    counts_order = counts.index
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=week_days_freq, order=counts_order, ax=ax)
    plt.xlabel('Booking Date Week Day')
    plt.ylabel('Count')
    plt.title('Booking Date Week Day Distribution')
    for i in range(counts.shape[0]):
        plt.text(i, counts[i]+200, f"{counts[i]/week_days_freq.shape[0]*100:0.2f}%", ha='center', fontsize=9.5)
    sns.despine()
    st.pyplot(fig)

    # Destination Country Distribution
    st.write("#### Destination Country Distribution")
    counts = user_data['country_destination'].value_counts()
    counts_order = counts.index
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=user_data['country_destination'], ax=ax)
    plt.xlabel('Destination Country')
    plt.ylabel('Count')
    plt.title('Destination Country Distribution')
    for i in range(counts.shape[0]):
        plt.text(i, counts[i]+1000, f"{counts[i]/user_data.shape[0]*100:0.2f}%", ha='center', fontsize=10)
    sns.despine()
    st.pyplot(fig)

    # Account Creation Date Frequency
    st.write("#### Account Creation Date Frequency")
    creation_dates = user_data['date_account_created'].value_counts().sort_index()
    creation_dates.index = pd.to_datetime(creation_dates.index)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.lineplot(x=creation_dates.index, y=creation_dates.values, ax=ax)
    plt.xlabel('Account Creation Date')
    plt.ylabel('Count')
    plt.title('Account Creation Date Frequency')
    anomalies = ['2011-09', '2012-09', '2013-09']
    for anomaly in anomalies:
        ax.axvline(pd.to_datetime(anomaly, format='%Y-%m'), color='#CB4335', linestyle='dashed', linewidth=1.1)
        ax.text(pd.to_datetime(anomaly, format='%Y-%m'), -14, anomaly, ha='center', color='#92140C')
    sns.despine()
    st.pyplot(fig)

    # Monthly Trends in Account Creation
   # st.write("#### Monthly Trends in Account Creation")
   # user_data['date_account_created'] = pd.to_datetime(user_data['date_account_created'])
   # user_data['year_month'] = user_data['date_account_created'].dt.to_period('M').astype(str)
   # monthly_counts = user_data.groupby('year_month').size().reset_index(name='counts')
   # fig = px.line(monthly_counts, x='year_month', y='counts', title='Monthly Trends in Account Creation')
   # st.plotly_chart(fig)

    # Monthly Trends in Account Creation
    st.write("#### Monthly Trends in Account Creation")
    user_data['date_account_created'] = pd.to_datetime(user_data['date_account_created'])
    user_data['year'] = user_data['date_account_created'].dt.year
    user_data['year_month'] = user_data['date_account_created'].dt.to_period('M').astype(str)
    monthly_counts = user_data.groupby('year_month').size().reset_index(name='counts')
    fig = go.Figure()
    
    years = sorted(user_data['year'].unique())
    for year in years:
        filtered_data = monthly_counts[monthly_counts['year_month'].str.startswith(str(year))]
        fig.add_trace(go.Scatter(
            x=filtered_data['year_month'],
            y=filtered_data['counts'],
            mode='lines+markers',
            name=str(year),
            visible=False
        ))
    
    # Make the first year trace visible by default
    if len(fig.data) > 0:
        fig.data[0].visible = True
    
    # Create dropdown buttons to toggle visibility of traces
    dropdown_buttons = [
        {
            'label': f'Year {year}',
            'method': 'update',
            'args': [{'visible': [year == int(trace.name) for trace in fig.data]}]
        }
        for year in years
    ]
    dropdown_buttons.append(
        {
            'label': 'Show All',
            'method': 'update',
            'args': [{'visible': [True] * len(fig.data)}]
        }
    )
    fig.update_layout(
        title='Monthly Trends in Account Creation',
        xaxis_title='Month',
        yaxis_title='Number of Accounts',
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig)

    # Monthly Trends in Account Creation for Users Traveling to the US
   # st.write("#### Monthly Trends in Account Creation for Users Traveling to the US")
   # us_travel_data = user_data[user_data['country_destination'] == 'US']
   # monthly_counts = us_travel_data.groupby('year_month').size().reset_index(name='counts')
   # fig = px.line(monthly_counts, x='year_month', y='counts', title='Monthly Trends for US Travelers')
   # st.plotly_chart(fig)

    # Monthly Trends in Account Creation for Users Traveling to the US
    st.write("#### Monthly Trends in Account Creation for Users Traveling to the US")
    user_data['date_account_created'] = pd.to_datetime(user_data['date_account_created'])
    user_data['year'] = user_data['date_account_created'].dt.year
    user_data['year_month'] = user_data['date_account_created'].dt.to_period('M').astype(str)
    us_travel_data = user_data[user_data['country_destination'] == 'US']
    monthly_counts = us_travel_data.groupby('year_month').size().reset_index(name='counts')
    fig = go.Figure()
    
    years = sorted(us_travel_data['year'].unique())
    for year in years:
        filtered_data = monthly_counts[monthly_counts['year_month'].str.startswith(str(year))]
        fig.add_trace(go.Scatter(
            x=filtered_data['year_month'],
            y=filtered_data['counts'],
            mode='lines+markers',
            name=str(year),
            visible=False
        ))
    
    if len(fig.data) > 0:
        fig.data[0].visible = True
    dropdown_buttons = [
        {
            'label': f'Year {year}',
            'method': 'update',
            'args': [{'visible': [year == int(trace.name) for trace in fig.data]}]
        }
        for year in years
    ]
    
    dropdown_buttons.append(
        {
            'label': 'Show All',
            'method': 'update',
            'args': [{'visible': [True] * len(fig.data)}]
        }
    )
    fig.update_layout(
        title='Monthly Trends in Account Creation for Users Traveling to the US',
        xaxis_title='Month',
        yaxis_title='Number of Accounts',
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig)

    # Destination Country Distribution Per Gender
    st.write("#### Destination Country Distribution Per Gender")
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.countplot(x='country_destination', hue='gender', data=user_data, ax=ax)
    plt.xlabel('Destination Country')
    plt.ylabel('Count')
    plt.title('Destination Country Distribution Per Gender')
    sns.despine()
    st.pyplot(fig)

    # Destination Country Distribution Per Age
    st.write("#### Destination Country Distribution Per Age (Ages 0 - 120)")
    filtered_data = user_data[(user_data['age'] >= 0) & (user_data['age'] <= 120)]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='country_destination', y='age', data=filtered_data, ax=ax)
    ax.set_xlabel('Destination Country')
    ax.set_ylabel('Age')
    ax.set_title('Destination Country Distribution Per Age')
    sns.despine()
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
    user_data['date_account_created'] = pd.to_datetime(user_data['date_account_created'])
    user_data['cohort_month'] = user_data['date_account_created'].dt.to_period('M')
    user_data['cohort_year'] = user_data['cohort_month'].dt.year
    cohort_counts = user_data.groupby(['cohort_month', 'country_destination']).size().unstack().fillna(0)
    cohort_percent = cohort_counts.divide(cohort_counts.sum(axis=1), axis=0) * 100
    cohort_percent.index = cohort_percent.index.astype(str)
    fig = go.Figure()
    
    # Add a trace for each year
    years = sorted(user_data['cohort_year'].unique())
    for year in years:
        filtered_data = cohort_percent[cohort_percent.index.str.startswith(str(year))]
        fig.add_trace(go.Heatmap(
            z=filtered_data.T.values,
            x=filtered_data.index,
            y=filtered_data.columns,
            colorscale='Blues',
            visible=False,
            name=str(year)
        ))
    
    if len(fig.data) > 0:
        fig.data[0].visible = True
    
    dropdown_buttons = [
        {
            'label': f'Year {year}',
            'method': 'update',
            'args': [{'visible': [year == int(trace.name) for trace in fig.data]},
                     {'title': f'Cohort Analysis by Destination Country for {year}'}]
        }
        for year in years
    ]
    
    dropdown_buttons.append(
        {
            'label': 'Show All',
            'method': 'update',
            'args': [{'visible': [True] * len(fig.data)},
                     {'title': 'Cohort Analysis by Destination Country (All Years)'}]
        }
    )
    
    # Update layout with dropdown menu
    fig.update_layout(
        title='Cohort Analysis by Destination Country',
        xaxis_title='Cohort Month',
        yaxis_title='Destination Country',
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig)

    # Whether Members Booked Per Gender
    st.write("#### Whether Members Booked Per Gender")
    booked_status = user_data['date_first_booking'].notna()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=booked_status, hue='gender', data=user_data, ax=ax)
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.title('Whether Members Booked Per Gender')
    sns.despine()
    st.pyplot(fig)

    # Whether Members Booked Per Gender
    st.write("#### Whether Members Booked Per Gender")
    booked_status = user_data['date_first_booking'].notna()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.countplot(x=booked_status, hue='signup_method', data=user_data, ax=ax)
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.title('Whether Members Booked Per Signup Method')
    sns.despine()
    st.pyplot(fig)

    # Monthly Booking Trends
    st.write("#### Monthly Booking Trends")
    user_data['date_first_booking'] = pd.to_datetime(user_data['date_first_booking'])
    user_data['year_month_booking'] = user_data['date_first_booking'].dt.to_period('M').astype(str)
    monthly_bookings = user_data.groupby('year_month_booking').size().reset_index(name='counts')
    monthly_bookings['counts'] = monthly_bookings['counts'].clip(upper=6000)
    fig = px.line(monthly_bookings, x='year_month_booking', y='counts', title='Monthly Booking Trends')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.title('Whether Members Booked Per Signup Method')
    sns.despine()
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
    st.write("### Search User ID")
    selected_test_id = st.selectbox("Select User ID", sub_whole_df['id'].unique())
    filtered_by_test_id = sub_whole_df[sub_whole_df['id'] == selected_test_id]
    st.write(f"### Country results for User ID: {selected_test_id}")
    st.write(filtered_by_test_id)

    # Provide a clickable link to Airbnb
    st.markdown('[Go to Airbnb website](https://www.airbnb.com/)')

    # Add extra information using expander
    with st.expander("Click here for more information about countries"):
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
    st.write("### Search Country")
    selected_lbl_encoder = st.selectbox("Select Country", sub_whole_df['country'].unique())
    filtered_by_lbl_encoder = sub_whole_df[sub_whole_df['country'] == selected_lbl_encoder]
    st.write(f"### User ID for Country {selected_lbl_encoder}")
    st.write(filtered_by_lbl_encoder)

    # Provide a clickable link to Airbnb
    st.markdown('[Go to Airbnb website](https://www.airbnb.com/)')

    # Add extra information using expander
    with st.expander("Click here for more information about countries"):
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
elif page == "3.0 Dropdown for User ID":
    dropdown_test_ids()
elif page == "4.0 Dropdown for Country":
    dropdown_lbl_encoder()
