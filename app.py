# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 22:12:38 2022

@author: Shayan
"""
import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import string
pd.options.mode.chained_assignment = None
import chatwords as cw
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px




REMOVE_PUNCT = string.punctuation
def del_puctuation(text):
    """Function to remove the punctuation"""
    return text.translate(str.maketrans('', '', REMOVE_PUNCT))


from nltk.corpus import stopwords
SW = set(stopwords.words('english'))
def del_stopwords(text):
    """Function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in SW])


from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {
    "N":wordnet.NOUN,
    "V":wordnet.VERB,
    "J":wordnet.ADJ,
    "R":wordnet.ADV
    }
#function for lemmatize words
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return (
        " ".join(
            [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) 
             for word, pos in pos_tagged_text]
            )
        )

#function to remove URLs
def del_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


#function to preprocess SMS Data
def wrangle(data):
    #changing data type of date column
    data["Date_Received"] = pd.to_datetime(data["Date_Received"]).dt.date
    #adding month name column to visualize sms recieved per month
    data["Month"] = pd.to_datetime(data["Date_Received"]).dt.month_name()
    #removing chat words by importing custom module chatwords
    #removing it first because data in chatwords are in uppercase
    #and contain stopwords too
    data["Message_body"]= (
        data["Message_body"]
        .apply(lambda text: cw.chat_words_conversion(text))
        )
    #changing the text into lower case
    data["Message_body"] = data["Message_body"].str.lower()
    #removing punctuation from messages
    data["Message_body"] = (
        data["Message_body"]
        .apply(lambda text: del_puctuation(text))
        )
    #removing stopwords
    data["Message_body"] = (
        data["Message_body"]
        .apply(lambda text: del_stopwords(text))
        )
    #lemmatize the words
    data["Message_body"] = (
        data["Message_body"]
        .apply(lambda text: lemmatize_words(text))
        )
    #removing url from messages
    data["Message_body"] = (
        data["Message_body"]
        .apply(lambda text: del_urls(text))
        )
    return data

#function to count repeatative words in message body
def count_words_df(data):
    counter = Counter(" ".join(data["Message_body"])
                      .split()).most_common(10)
    counter_df = pd.DataFrame(counter, columns = ['Word', 'Frequency'])
    return counter_df

#function to count number of messages in a month
def sms_count(data):
    monthly_sms_count = (
        data.groupby("Month")["Message_body"]
        .count().sort_values(ascending=False)
        ).to_frame().reset_index()
    monthly_sms_count.rename(columns={"Message_body":"Frequency"},inplace=True)
    return monthly_sms_count
    

def main():
    #setting wide layout
    st. set_page_config(layout="wide")
    st.title("SMS Text Preprocessor App")
    #uploading file
    file = st.file_uploader("Upload CSV File")
    if file is not None:
        full_df = pd.read_csv(file,encoding='unicode_escape',index_col="S. No.")
        df = wrangle(full_df)
        #selection box to select SPAM or NON SPAM sms
        label = st.selectbox("Select Label",df["Label"].unique())
        button = st.button("Show")
        if button:
            #masking data
            label_data = df[df["Label"] == label]
            
            #Generating WordCloud
            c_words = " ".join(review for review in label_data.Message_body)
            w_cloud = (
                WordCloud(background_color="white")
                .generate(c_words)
                )
            fig, ax = plt.subplots(figsize = (8, 4))
            ax.imshow(w_cloud)
            plt.axis("off")
            st.pyplot(fig)
            #End of WordCloud
            
            #Streamlit Column Layout
            col1, col2 = st.columns(2,gap="small")
            
            #Col 1
            with col1:
                #10 Most Common Words
                st.subheader("Bar Graph of 10 Most Common Words")
                c_df = count_words_df(label_data)
                #Plotting
                bar_fig = (
                    px.bar(c_df,x="Frequency",y="Word"
                           ,orientation='h',color="Word"))
                st.plotly_chart(bar_fig,use_container_width=True)
            
            #Col 2
            with col2:
                #Monthly SMS Frequency
                st.subheader("Line plot of Messages Count by Month")
                m_df = sms_count(label_data)
                #Plotting
                line_fig = px.line(m_df, x="Month",y="Frequency")
                st.plotly_chart(line_fig,use_container_width=True)
                
            st.markdown(
                "<h6 style='text-align: center; color: black;'>Made with ❤ Shayan Khanani</h6>", 
                unsafe_allow_html=True
                )
                
if __name__ == "__main__":
    main()
