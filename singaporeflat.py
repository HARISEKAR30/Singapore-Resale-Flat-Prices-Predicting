import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import os
import joblib
from PIL import Image

#importing the csv file to encode catgorical values
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\DATA SCIENCE\PROJECT\SINGAPORE SALES PRED\processed_singapore_flat.csv")

#encoding the town values like label encoding 
town_uni = sorted(list(df.town.unique()))

town_dict = {town: ind for ind, town in enumerate(town_uni)}


def town_mapping(town):
    return town_dict[town]
   




#encoding the flat type values
flat_type_uni = sorted(list(df.flat_type.unique()))

flat_type_dict = {flat_type: ind for ind, flat_type in enumerate(flat_type_uni)}


def flat_type_mapping(flat_type):
    return flat_type_dict[flat_type]


#encoding the street_name values

street_name_unique = sorted(list(df.street_name.unique()))

street_name_dict ={street_name: ind for ind, street_name in enumerate(street_name_unique)}

def street_name_mapping(street_name):
    return street_name_dict[street_name]


#encoding the flat_model values

flat_model_unique = sorted(list(df.flat_model.unique()))

flat_model_dict = {flat_model: ind for ind, flat_model in enumerate(flat_model_unique)}



def flat_model_mapping(flat_model):
    return flat_model_dict[flat_model]




#function for predicting the flat resale_price

def predict_resale_price(town,flat_type,block,street_name,floor_area_sqm,
                        flat_model,lease_commence_date,storey_range_start,
                        storey_range_end,resale_year,resale_month):
    town = town_mapping(town)
    flat_type = flat_type_mapping(flat_type)
    block = float(block)
    street_name = street_name_mapping(street_name)
    floor_area_sqm = float(floor_area_sqm)
    flat_model = flat_model_mapping(flat_model)
    lease_commence_date = int(lease_commence_date)
    storey_range_start = np.log(storey_range_start)
    storey_range_end = np.log(storey_range_end)
    resale_year = int(resale_year)
    resale_month = int(resale_month) 

    user_feed = np.array([[town,flat_type,block,street_name,floor_area_sqm,
                            flat_model,lease_commence_date,storey_range_start,
                            storey_range_end,resale_year,resale_month]]) 
    print(user_feed)
    
    # loading the trained model
    file_path = "D:/vscode/Flatprice_model.pkl"
    
    with open(file_path,"rb") as f:
        regg_model =  joblib.load(f)
    
    y_pred = regg_model.predict(user_feed)
    y_pred_actual = np.exp(y_pred[0])
    pred_price = round(y_pred_actual)

    return pred_price


st.set_page_config(layout="wide")

st.title("SINGAPORE RESALE FLAT PRICES PREDICTING")
st.write("")

with st.sidebar:
    select= option_menu("MAIN MENU",["Home", "Price Prediction", "About"])

if select == "Home":
    st.write("Our project aims to develop a machine learning model capable of accurately predicting resale flat prices in Singapore. Housing affordability and market transparency are critical concerns for both buyers and sellers in Singapore's real estate market. By leveraging historical transaction data and advanced machine learning techniques, our model provides valuable insights into future resale flat prices, aiding buyers, sellers, and real estate professionals in making informed decisions.")

elif select == "Price Prediction":

    col1,col2= st.columns(2)
    with col1:

        resale_year= st.selectbox("Select Resale Year",[1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
                                                    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                                                    2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
                                                    2023, 2024])


        resale_month= st.selectbox("Select Resale Month",[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])                                         
                                                    
        lease_commence_date= st.selectbox("Select Lease Commence Year",[1977, 1976, 1978, 1979, 1984, 1980, 1985, 1981, 1982, 1986, 1972,
                                                    1983, 1973, 1969, 1975, 1971, 1974, 1967, 1970, 1968, 1988, 1987,
                                                    1989, 1990, 1992, 1993, 1994, 1991, 1995, 1996, 1997, 1998, 1999,
                                                    2000, 2001, 1966, 2002, 2006, 2003, 2005, 2004, 2008, 2007, 2009,
                                                    2010, 2012, 2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022,
                                                    2020])
        
        town= st.selectbox("Select Town", ['Ang mo kio', 'Bedok', 'Bishan', 'Bukit batok', 'Bukit merah',
                                                'Bukit timah', 'Central area', 'Choa chu kang', 'Clementi',
                                                'Geylang', 'Hougang', 'Jurong east', 'Jurong west',
                                                'Kallang/whampoa', 'Marine parade', 'Queenstown', 'Sengkang',
                                                'Serangoon', 'Tampines', 'Toa payoh', 'Woodlands', 'Yishun',
                                                'Lim chu kang', 'Sembawang', 'Bukit panjang', 'Pasir ris',
                                                'Punggol'])
        
        flat_type= st.selectbox("Select Flat Type", ['1 room', '3 room', '4 room', '5 room', '2 room', 'Executive',
                                                        'Multi generation'])
        
        

        flat_model= st.selectbox("Select Flat Model", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                        'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                        'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                        'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                        'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])
        
    with col2:

        floor_area_sqm= st.number_input("Enter  Floor Area(sqm) (Min: 31 / Max: 280")

        storey_range_start= st.number_input("Enter Starting number of  Storey (Min:1/ Max:49)")

        storey_range_end= st.number_input("Enter ending number of storey (Min:3/ Max:51)")

        block = st.slider('Select Block:', 1, 980)

        street_name = st.selectbox("Select Street", ['Admiralty dr', 'Admiralty link', 'Ah hood rd', 'Alexandra rd',
                                                    'Aljunied ave 2', 'Aljunied cres', 'Aljunied rd',
                                                    'Anchorvale cres', 'Anchorvale dr', 'Anchorvale lane',
                                                    'Anchorvale link', 'Anchorvale rd', 'Anchorvale st',
                                                    'Ang mo kio ave 1', 'Ang mo kio ave 10', 'Ang mo kio ave 2',
                                                    'Ang mo kio ave 3', 'Ang mo kio ave 4', 'Ang mo kio ave 5',
                                                    'Ang mo kio ave 6', 'Ang mo kio ave 8', 'Ang mo kio ave 9',
                                                    'Ang mo kio st 11', 'Ang mo kio st 21', 'Ang mo kio st 31',
                                                    'Ang mo kio st 32', 'Ang mo kio st 44', 'Ang mo kio st 51',
                                                    'Ang mo kio st 52', 'Ang mo kio st 61', 'Bain st', 'Balam rd',
                                                    'Bangkit rd', 'Beach rd', 'Bedok ctrl', 'Bedok nth ave 1',
                                                    'Bedok nth ave 2', 'Bedok nth ave 3', 'Bedok nth ave 4',
                                                    'Bedok nth rd', 'Bedok nth st 1', 'Bedok nth st 2',
                                                    'Bedok nth st 3', 'Bedok nth st 4', 'Bedok reservoir cres',
                                                    'Bedok reservoir rd', 'Bedok reservoir view', 'Bedok sth ave 1',
                                                    'Bedok sth ave 2', 'Bedok sth ave 3', 'Bedok sth rd',
                                                    'Bendemeer rd', 'Beo cres', 'Bishan st 11', 'Bishan st 12',
                                                    'Bishan st 13', 'Bishan st 22', 'Bishan st 23', 'Bishan st 24',
                                                    'Boon keng rd', 'Boon lay ave', 'Boon lay dr', 'Boon lay pl',
                                                    'Boon tiong rd', 'Bright hill dr', 'Bt batok ctrl',
                                                    'Bt batok east ave 3', 'Bt batok east ave 4',
                                                    'Bt batok east ave 5', 'Bt batok east ave 6', 'Bt batok st 11',
                                                    'Bt batok st 21', 'Bt batok st 22', 'Bt batok st 24',
                                                    'Bt batok st 25', 'Bt batok st 31', 'Bt batok st 32',
                                                    'Bt batok st 33', 'Bt batok st 34', 'Bt batok st 51',
                                                    'Bt batok st 52', 'Bt batok west ave 2', 'Bt batok west ave 4',
                                                    'Bt batok west ave 5', 'Bt batok west ave 6',
                                                    'Bt batok west ave 7', 'Bt batok west ave 8',
                                                    'Bt batok west ave 9', 'Bt merah ctrl', 'Bt merah lane 1',
                                                    'Bt merah view', 'Bt panjang ring rd', 'Bt purmei rd',
                                                    'Buangkok cres', 'Buangkok green', 'Buangkok link',
                                                    'Buangkok sth farmway 1', 'Buffalo rd', "C'wealth ave",
                                                    "C'wealth ave west", "C'wealth cl", "C'wealth cres", "C'wealth dr",
                                                    'Cambridge rd', 'Canberra cres', 'Canberra link', 'Canberra rd',
                                                    'Canberra st', 'Canberra walk', 'Cantonment cl', 'Cantonment rd',
                                                    'Cashew rd', 'Cassia cres', 'Chai chee ave', 'Chai chee dr',
                                                    'Chai chee rd', 'Chai chee st', 'Chander rd', 'Changi village rd',
                                                    'Chin swee rd', 'Choa chu kang ave 1', 'Choa chu kang ave 2',
                                                    'Choa chu kang ave 3', 'Choa chu kang ave 4',
                                                    'Choa chu kang ave 5', 'Choa chu kang ave 7', 'Choa chu kang cres',
                                                    'Choa chu kang ctrl', 'Choa chu kang dr', 'Choa chu kang loop',
                                                    'Choa chu kang nth 5', 'Choa chu kang nth 6',
                                                    'Choa chu kang nth 7', 'Choa chu kang st 51',
                                                    'Choa chu kang st 52', 'Choa chu kang st 53',
                                                    'Choa chu kang st 54', 'Choa chu kang st 62',
                                                    'Choa chu kang st 64', 'Circuit rd', 'Clarence lane',
                                                    'Clementi ave 1', 'Clementi ave 2', 'Clementi ave 3',
                                                    'Clementi ave 4', 'Clementi ave 5', 'Clementi ave 6',
                                                    'Clementi st 11', 'Clementi st 12', 'Clementi st 13',
                                                    'Clementi st 14', 'Clementi west st 1', 'Clementi west st 2',
                                                    'Compassvale bow', 'Compassvale cres', 'Compassvale dr',
                                                    'Compassvale lane', 'Compassvale link', 'Compassvale rd',
                                                    'Compassvale st', 'Compassvale walk', 'Corporation dr',
                                                    'Crawford lane', 'Dakota cres', 'Dawson rd', 'Delta ave',
                                                    'Depot rd', 'Dorset rd', 'Dover cl east', 'Dover cres', 'Dover rd',
                                                    'East coast rd', 'Edgedale plains', 'Edgefield plains', 'Elias rd',
                                                    'Empress rd', 'Eunos cres', 'Eunos rd 5', 'Everton pk', 'Fajar rd',
                                                    'Farrer pk rd', 'Farrer rd', 'Fernvale lane', 'Fernvale link',
                                                    'Fernvale rd', 'Fernvale st', 'French rd', 'Gangsa rd',
                                                    'Geylang bahru', 'Geylang east ave 1', 'Geylang east ave 2',
                                                    'Geylang east ctrl', 'Geylang serai', 'Ghim moh link',
                                                    'Ghim moh rd', 'Gloucester rd', 'Haig rd', 'Havelock rd',
                                                    'Henderson cres', 'Henderson rd', 'Hillview ave', 'Ho ching rd',
                                                    'Holland ave', 'Holland cl', 'Holland dr', 'Hougang ave 1',
                                                    'Hougang ave 10', 'Hougang ave 2', 'Hougang ave 3',
                                                    'Hougang ave 4', 'Hougang ave 5', 'Hougang ave 6', 'Hougang ave 7',
                                                    'Hougang ave 8', 'Hougang ave 9', 'Hougang ctrl', 'Hougang st 11',
                                                    'Hougang st 21', 'Hougang st 22', 'Hougang st 31', 'Hougang st 32',
                                                    'Hougang st 51', 'Hougang st 52', 'Hougang st 61', 'Hougang st 91',
                                                    'Hougang st 92', 'Hoy fatt rd', 'Hu ching rd', 'Indus rd',
                                                    'Jelapang rd', 'Jelebu rd', 'Jellicoe rd', 'Jln bahagia',
                                                    'Jln batu', 'Jln berseh', 'Jln bt ho swee', 'Jln bt merah',
                                                    'Jln damai', 'Jln dua', 'Jln dusun', 'Jln kayu', 'Jln klinik',
                                                    'Jln kukoh', "Jln ma'mor", 'Jln membina', 'Jln membina barat',
                                                    'Jln pasar baru', 'Jln rajah', 'Jln rumah tinggi', 'Jln teck whye',
                                                    'Jln tenaga', 'Jln tenteram', 'Jln tiga', 'Joo chiat rd',
                                                    'Joo seng rd', 'Jurong east ave 1', 'Jurong east st 13',
                                                    'Jurong east st 21', 'Jurong east st 24', 'Jurong east st 31',
                                                    'Jurong east st 32', 'Jurong west ave 1', 'Jurong west ave 3',
                                                    'Jurong west ave 5', 'Jurong west ctrl 1', 'Jurong west ctrl 3',
                                                    'Jurong west st 24', 'Jurong west st 25', 'Jurong west st 41',
                                                    'Jurong west st 42', 'Jurong west st 51', 'Jurong west st 52',
                                                    'Jurong west st 61', 'Jurong west st 62', 'Jurong west st 64',
                                                    'Jurong west st 65', 'Jurong west st 71', 'Jurong west st 72',
                                                    'Jurong west st 73', 'Jurong west st 74', 'Jurong west st 75',
                                                    'Jurong west st 81', 'Jurong west st 91', 'Jurong west st 92',
                                                    'Jurong west st 93', 'Kallang bahru', 'Kang ching rd',
                                                    'Keat hong cl', 'Keat hong link', 'Kelantan rd', 'Kent rd',
                                                    'Kg arang rd', 'Kg bahru hill', 'Kg kayu rd', 'Kim cheng st',
                                                    'Kim keat ave', 'Kim keat link', 'Kim pong rd', 'Kim tian pl',
                                                    'Kim tian rd', "King george's ave", 'Klang lane', 'Kreta ayer rd',
                                                    'Lengkok bahru', 'Lengkong tiga', 'Lim chu kang rd', 'Lim liak st',
                                                    'Lompang rd', 'Lor 1 toa payoh', 'Lor 1a toa payoh',
                                                    'Lor 2 toa payoh', 'Lor 3 geylang', 'Lor 3 toa payoh',
                                                    'Lor 4 toa payoh', 'Lor 5 toa payoh', 'Lor 6 toa payoh',
                                                    'Lor 7 toa payoh', 'Lor 8 toa payoh', 'Lor ah soo', 'Lor lew lian',
                                                    'Lor limau', 'Lower delta rd', 'Macpherson lane', 'Margaret dr',
                                                    'Marine cres', 'Marine dr', 'Marine parade ctrl', 'Marine ter',
                                                    'Marsiling cres', 'Marsiling dr', 'Marsiling lane', 'Marsiling rd',
                                                    'Marsiling rise', 'Mcnair rd', 'Mei ling st', 'Moh guan ter',
                                                    'Montreal dr', 'Montreal link', 'Moulmein rd', 'New mkt rd',
                                                    'New upp changi rd', 'Nile rd', 'Nth bridge rd', 'Old airport rd',
                                                    'Outram hill', 'Outram pk', 'Owen rd', 'Pandan gdns',
                                                    'Pasir ris dr 1', 'Pasir ris dr 10', 'Pasir ris dr 3',
                                                    'Pasir ris dr 4', 'Pasir ris dr 6', 'Pasir ris st 11',
                                                    'Pasir ris st 12', 'Pasir ris st 13', 'Pasir ris st 21',
                                                    'Pasir ris st 41', 'Pasir ris st 51', 'Pasir ris st 52',
                                                    'Pasir ris st 53', 'Pasir ris st 71', 'Pasir ris st 72',
                                                    'Paya lebar way', 'Pending rd', 'Petir rd', 'Pine cl', 'Pipit rd',
                                                    'Potong pasir ave 1', 'Potong pasir ave 2', 'Potong pasir ave 3',
                                                    'Punggol ctrl', 'Punggol dr', 'Punggol east', 'Punggol field',
                                                    'Punggol field walk', 'Punggol pl', 'Punggol rd', 'Punggol walk',
                                                    'Punggol way', 'Queen st', "Queen's cl", "Queen's rd", 'Queensway',
                                                    'Race course rd', 'Redhill cl', 'Redhill lane', 'Redhill rd',
                                                    'Rivervale cres', 'Rivervale dr', 'Rivervale st', 'Rivervale walk',
                                                    'Rochor rd', 'Rowell rd', 'Sago lane', 'Saujana rd', 'Segar rd',
                                                    'Selegie rd', 'Seletar west farmway 6', 'Sembawang cl',
                                                    'Sembawang cres', 'Sembawang dr', 'Sembawang rd',
                                                    'Sembawang vista', 'Sembawang way', 'Seng poh rd', 'Sengkang ctrl',
                                                    'Sengkang east ave', 'Sengkang east rd', 'Sengkang east way',
                                                    'Sengkang west ave', 'Sengkang west way', 'Senja link', 'Senja rd',
                                                    'Serangoon ave 1', 'Serangoon ave 2', 'Serangoon ave 3',
                                                    'Serangoon ave 4', 'Serangoon ctrl', 'Serangoon ctrl dr',
                                                    'Serangoon nth ave 1', 'Serangoon nth ave 2',
                                                    'Serangoon nth ave 3', 'Serangoon nth ave 4', 'Short st',
                                                    'Shunfu rd', 'Silat ave', 'Simei lane', 'Simei rd', 'Simei st 1',
                                                    'Simei st 2', 'Simei st 4', 'Simei st 5', 'Sims ave', 'Sims dr',
                                                    'Sims pl', 'Sin ming ave', 'Sin ming rd', 'Smith st',
                                                    'Spottiswoode pk rd', "St. george's lane", "St. george's rd",
                                                    'Stirling rd', 'Strathmore ave', 'Sumang lane', 'Sumang link',
                                                    'Sumang walk', 'Tah ching rd', 'Taman ho swee', 'Tampines ave 1',
                                                    'Tampines ave 4', 'Tampines ave 5', 'Tampines ave 7',
                                                    'Tampines ave 8', 'Tampines ave 9', 'Tampines ctrl 1',
                                                    'Tampines ctrl 7', 'Tampines ctrl 8', 'Tampines st 11',
                                                    'Tampines st 12', 'Tampines st 21', 'Tampines st 22',
                                                    'Tampines st 23', 'Tampines st 24', 'Tampines st 32',
                                                    'Tampines st 33', 'Tampines st 34', 'Tampines st 41',
                                                    'Tampines st 42', 'Tampines st 43', 'Tampines st 44',
                                                    'Tampines st 45', 'Tampines st 61', 'Tampines st 71',
                                                    'Tampines st 72', 'Tampines st 81', 'Tampines st 82',
                                                    'Tampines st 83', 'Tampines st 84', 'Tampines st 86',
                                                    'Tampines st 91', 'Tanglin halt rd', 'Tao ching rd',
                                                    'Teban gdns rd', 'Teck whye ave', 'Teck whye cres',
                                                    'Teck whye lane', 'Telok blangah cres', 'Telok blangah dr',
                                                    'Telok blangah hts', 'Telok blangah rise', 'Telok blangah st 31',
                                                    'Telok blangah way', 'Tessensohn rd', 'Tg pagar plaza',
                                                    'Tiong bahru rd', 'Toa payoh ctrl', 'Toa payoh east',
                                                    'Toa payoh nth', 'Toh guan rd', 'Toh yi dr', 'Towner rd',
                                                    'Ubi ave 1', 'Upp aljunied lane', 'Upp boon keng rd',
                                                    'Upp cross st', 'Upp serangoon cres', 'Upp serangoon rd',
                                                    'Upp serangoon view', 'Veerasamy rd', 'Waterloo st',
                                                    'Wellington circle', 'West coast dr', 'West coast rd',
                                                    'Whampoa dr', 'Whampoa rd', 'Whampoa sth', 'Whampoa west',
                                                    'Woodlands ave 1', 'Woodlands ave 3', 'Woodlands ave 4',
                                                    'Woodlands ave 5', 'Woodlands ave 6', 'Woodlands ave 9',
                                                    'Woodlands circle', 'Woodlands cres', 'Woodlands ctr rd',
                                                    'Woodlands dr 14', 'Woodlands dr 16', 'Woodlands dr 40',
                                                    'Woodlands dr 42', 'Woodlands dr 44', 'Woodlands dr 50',
                                                    'Woodlands dr 52', 'Woodlands dr 53', 'Woodlands dr 60',
                                                    'Woodlands dr 62', 'Woodlands dr 70', 'Woodlands dr 71',
                                                    'Woodlands dr 72', 'Woodlands dr 73', 'Woodlands dr 75',
                                                    'Woodlands ring rd', 'Woodlands rise', 'Woodlands st 11',
                                                    'Woodlands st 13', 'Woodlands st 31', 'Woodlands st 32',
                                                    'Woodlands st 41', 'Woodlands st 81', 'Woodlands st 82',
                                                    'Woodlands st 83', 'Yishun ave 1', 'Yishun ave 11', 'Yishun ave 2',
                                                    'Yishun ave 3', 'Yishun ave 4', 'Yishun ave 5', 'Yishun ave 6',
                                                    'Yishun ave 7', 'Yishun ave 9', 'Yishun ctrl', 'Yishun ctrl 1',
                                                    'Yishun ring rd', 'Yishun st 11', 'Yishun st 20', 'Yishun st 21',
                                                    'Yishun st 22', 'Yishun st 31', 'Yishun st 41', 'Yishun st 43',
                                                    'Yishun st 51', 'Yishun st 61', 'Yishun st 71', 'Yishun st 72',
                                                    'Yishun st 81', 'Yuan ching rd', 'Yung an rd', 'Yung ho rd',
                                                    'Yung kuang rd', 'Yung loh rd', 'Yung ping rd', 'Yung sheng rd',
                                                    'Zion rd'])


    button= st.button("Predict the Price", use_container_width= True)

    if button:

            
        predicted_price= predict_resale_price(town,flat_type,block,street_name,floor_area_sqm,
                                        flat_model, lease_commence_date,storey_range_start,
                                        storey_range_end, resale_year,resale_month)

        st.write("## :green[**The Predicted Price is :**]",predicted_price)


elif select == "About":

    st.header(":blue[Data Collection and Preprocessing:]")
    st.write("Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.")

    st.header(":blue[Feature Engineering:]")
    st.write("Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.")
    
    st.header(":blue[Model Selection and Training:]")
    st.write("Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.")

    st.header(":blue[Model Evaluation:]")
    st.write("Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.")

    st.header(":blue[Streamlit Web Application:]")
    st.write("Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.")

    st.header(":blue[Deployment on Render:]")
    st.write("Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.")
    
    st.header(":blue[Testing and Validation:]")
    st.write("Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.")