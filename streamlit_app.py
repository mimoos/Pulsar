import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path

import altair as alt
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Pulsar',
    page_icon='Pulsar.png', # This is an emoji shortcode. Could be a URL too.
)

Pulsar_data = pd.read_csv('data/Pulsar.csv')

y = Pulsar_data['Class']
X = Pulsar_data.drop(['Class'], axis = 1)

X_Features = X.columns

#Model = SGDClassifier(loss="modified_huber", penalty='l2', alpha=1e-3, random_state=42)
#Model = CalibratedClassifierCV(base_model)
Model = make_pipeline(StandardScaler(), LogisticRegression())
#Model = LogisticRegression()
#Model = SVC(probability=True)
Model.fit(X, y)

Pulsar_data

data = [None] * len(X_Features)
data[0] = Mean_Integrated_value = st.number_input('Insert Mean of Observations')
data[1] = SD_value = st.number_input('Insert Standard Deviation of Observations')
data[2] = EK_value = st.number_input('Insert Excess Kurtosis of Observations')
data[3] = Skewness_value = st.number_input('Insert Skewness of Observations')
data[4] = Mean_DMSNR_Curve_value = st.number_input('Insert Mean of DM SNR Curve of Obersations')
data[5] = SD_DMSNR_Curve_value = st.number_input('Insert Standard Deviation of DM SNR Curve of Obersations')
data[6] = EK_DMSNR_Curve_value = st.number_input('Insert Excess Kurtosis of DM SNR Curve of Obersations')
data[7] = Skewness_DMSNR_Curve_value = st.number_input('Insert Skewness of DM SNR Curve of Obersations')

#data = pd.DataFrame([data], columns = X_Features)
data_number = st.number_input('Insert the test number')
data = pd.DataFrame([Pulsar_data.iloc[int(data_number)]])
data = data.drop(['Class'], axis = 1)
data
Pulsar_button = st.button('Predict if it is a Pulsar')

if Pulsar_button == 1:
    y_pred = Model.predict(data)
    y_prob = np.max(Model.predict_proba(data))
    if y_pred == 0:
        st.write('The data entered of the object is not a pulsar, with the probability of', '%.02f' % (y_prob*100), '%.')
    else:
        st.write('The data entered of the object is a pulsar, with the probability of', '%.02f' % (y_prob*100), '%.')

# Predict on the Test Data
y_pred = Model.predict(X)

# Generate the confusion matrix
cm = confusion_matrix(y, y_pred)

cm
# Create a Confusion Matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

cr = classification_report(y, y_pred, target_names='Class')


# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
