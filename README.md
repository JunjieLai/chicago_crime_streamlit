# Shadows in the Sun - A guilty secret in a busy city

### This is the website for the final project.
![picture](https://github.com/Fairy-Rong/Final-project/blob/main/skyline.jpg?raw=true)

## Introduction
Crime in Chicago has always been an issue well worth exploring, and since many of our students have a strong desire to study in the United States, understanding how to protect yourself is even more of a meaningful topic. This repository is based on the data set collected by Chicago police to do data analysis as well as streamlit app.

## Team
[Yerong Wu](https://github.com/Fairy-Rong) & [Junjie Lai](https://github.com/JunjieLai)

## Dataset Description
[Chicago Crime Dataset](https://www.kaggle.com/datasets/chicago/chicago-crime?search=crime)

This dataset reflects reported incidents of crime (with the exception of murders where data exists for each victim) that occurred in the City of Chicago from 2012 to present. Dataset contains type of crime, date, location information, block, street, arrest or not, and latitude and longitude information.

## Presentation
[Presentation](https://github.com/Fairy-Rong/Final-project/blob/main/Team-14.pdf)

## Data App
[Streamlit Demo](https://fairy-rong-final-project-app-xm32au.streamlitapp.com/)

## The functions in the App
* 📈 Rolling sum of specific case types
* ⏱ Distribution of cases in different time periods
* 🏠 Temporal analysis of crime rates by type and location
* 🗺 Interactive Map with Location Search Function
* 🤖 Artificial Intelligence Dialogue based on GPT-3

## Repository Structure

| **Folder/Code** | **Content**                                              |
| :---------- | ------------------------------------------------------------ |
| .streamlit  | Contains the confiq.toml to set certain design parameters    |
| data        | Contains the scraped (and cleaned) Chicago Crime data in CSV format |
| apps        | Contains all the pages in streamlit app                      |
| geojson     | Contains GeoJson of Chicago administrative district division |
| multiapp.py | The python file used to implement multi-page functionality   |
| app.py      | The python file to run                                       |
|requirements.txt|Contains all requirements (necessary for Streamlit sharing)|
