import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
features = st.container()
model = st.container()

st.set_option('deprecation.showPyplotGlobalUse', False)

with header:
    # Define title text
    title_text = "Welcome to my cool Data Science project about running!"
    # Display centered title
    st.markdown(f"<h1 style='text-align: center'>{title_text}</h1>", unsafe_allow_html=True)
    
    st.markdown(
                '''
                This is a very simple "just to try it all part together" progect with a little bit 
                of EDA and basic model training.  
                My project provides valuable insights for runners looking to improve their training strategies, 
                and demonstrates the potential of data science in sports analysis.
                '''
                )

with dataset:
    st.header('Here is some information about my dataset')
    st.markdown(
                '''
                I decided to use my own running results over the past few years 
                to create a dataset for EDA and model training.  
                I chose only some of the many features that you can analyze to get some better
                insights into you performance.
                '''
                )
    
    st.write("<br>", unsafe_allow_html=True)
    st.subheader('Dataframe head:')
    df = pd.read_csv('data/my_runs.csv')
    st.write(df.head())
    st.write("<br>", unsafe_allow_html=True)

    
    st.subheader('Let\'s see some summury statistics:')
    st.write(df.describe())
    st.write("<br>", unsafe_allow_html=True)

   
    st.subheader('Some visualizations:')
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8,6))
    # Plot the distribution of data
    sns.histplot(df['Average Heart Rate'], 
                ax=ax,
                bins=15
                )
    # Create a more descriptive x axis label
    ax.set(xlabel="Average Heart Rate",
        ylabel='Counts',
        #xlim=(0,10),
        #ylim=(0,55),
        title="Average Heart Rate",
        )
    # Add vertical lines for the median and mean
    ax.axvline(x=df['Average Heart Rate'].median(), color='m', label='Median', linestyle='--', linewidth=2)
    ax.axvline(x=df['Average Heart Rate'].mean(), color='r', label='Mean', linestyle='-', linewidth=2)
    # Show the legend and plot the data
    ax.legend()
    st.pyplot()

    st.write("<br>", unsafe_allow_html=True)

    # Display a regression plot 
    fig, ax = plt.subplots(figsize=(8,6))
    sns.regplot(data=df,
         x='Pace',
         y='Average Heart Rate',
         marker='o',
         color='m',
         #x_bins=10,
         fit_reg=True,
         order=1
         )
    ax.set(xlabel="Pace",
        ylabel='Average Heart Rate',
        #xlim=(0,10),
        #ylim=(0,55),
        title="Average Heart Rate vs Pace",
        )     
    st.pyplot()

    st.write("<br>", unsafe_allow_html=True)

    # Create a jointplot similar to the JointGrid 
    fig, ax = plt.subplots(figsize=(8,6))
    sns.jointplot(x="Average Cadence",
            y="Average Heart Rate",
            kind='reg', #kind='resid'
            data=df,
            order=1
            )
    ax.set(xlabel="Average Cadence",
        ylabel='Average Heart Rate',
        #xlim=(0,10),
        #ylim=(0,55),
        title="Average Heart Rate vs Cadence",
        )
    st.pyplot()

    st.write("<br>", unsafe_allow_html=True)
    

with features:
    st.header('This is some information about features of my dataset')
    st.text("List of features in my dataset to use for model training:")
    st.table(df.columns)


with model:
    st.header('Model training')
    st.text('In this part we will train the model')
    
    sel_col, disp_col = st.columns(2)   

    max_depth = sel_col.slider('What should be the max_depth of the model?', 
                                min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees there will be?', options=[5, 10, 20, 30], index=0)

    input_feature = sel_col.text_input('Which feature should we use as an input?', 'Pace')

    # Model

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = df[[input_feature]]
    y = df[['Average Heart Rate']]

    regr.fit(X,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is: ')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is: ')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R score of the model is: ')
    disp_col.write(r2_score(y, prediction)) 


    
